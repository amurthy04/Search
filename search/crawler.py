"""High-throughput multi-process asyncio web crawler.

Architecture (v2):
- N worker processes; each owns the hosts whose hash(site) % N == worker id (so URL dedup,
  robots cache and per-host politeness are all process-local; no coordinator needed).
- Links discovered for foreign hosts are batched and routed to the owner over mp.Queues.
- Per worker: aiohttp + uvloop, keep-alive pool, pure-async dnspython resolver with cache
  (aiodns/c-ares blocked the loop ~2.4ms per lookup; see bench/dns_bench.py).
- selectolax (lexbor) parsing, ada-url normalization, 64-bit hashed seen-set.
- Breadth-first across hosts with per-host caps so the corpus is diverse.
- Per-request stage timings (dns, connect, ttfb, download, parse) recorded in every page.
- Output: zstd JSONL shards per worker.
"""
from __future__ import annotations

import asyncio
import collections
import heapq
import multiprocessing as mp
import os
import re
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

import aiohttp
import ada_url
import orjson
import tldextract
import xxhash
import zstandard as zstd
from protego import Protego
from selectolax.lexbor import LexborHTMLParser

try:
    import uvloop  # type: ignore
except ImportError:  # pragma: no cover
    uvloop = None

USER_AGENT = "SearchBot/0.2 (+https://github.com/amurthy04/Search; research crawler)"
BINARY_EXT = re.compile(
    r"\.(?:jpe?g|png|gif|webp|svg|ico|bmp|tiff?|mp[34]|m4[av]|mov|avi|mkv|webm|wav|ogg|flac|"
    r"pdf|docx?|xlsx?|pptx?|zip|gz|bz2|xz|7z|rar|tar|dmg|exe|msi|apk|iso|css|js|json|xml|rss|atom|woff2?|ttf|eot)$",
    re.I,
)
TRACKING = ("utm_", "fbclid", "gclid", "ref=", "ref_", "sessionid", "phpsessid", "sid=", "mc_cid", "mc_eid")
STRIP_TAGS = ["script", "style", "noscript", "svg", "canvas", "template", "iframe", "nav", "footer", "header", "aside", "form"]
_ws = re.compile(r"[ \t\r\f\v]+")
_nl = re.compile(r"\n\s*\n+")
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
_site_cache: dict[str, str] = {}


def site_key(host: str) -> str:
    k = _site_cache.get(host)
    if k is None:
        e = _extract(host)
        k = f"{e.domain}.{e.suffix}" if e.suffix else host
        if len(_site_cache) < 2_000_000:
            _site_cache[host] = k
    return k


def _finish(u: "ada_url.URL") -> Optional[str]:
    """Normalize an ada URL object to a canonical string (no fragment, tracking params dropped)."""
    proto = u.protocol
    if proto != "https:" and proto != "http:":
        return None
    host = u.hostname
    if not host or "." not in host:
        return None
    path = u.pathname or "/"
    if BINARY_EXT.search(path) or len(path) > 512:
        return None
    q = u.search
    if q:
        if len(q) > 512:
            return None
        if any(t in q.lower() for t in TRACKING):
            parts = [p for p in q[1:].split("&") if p and not p.lower().startswith(TRACKING)]
            q = "?" + "&".join(parts) if parts else ""
    port = u.port
    netloc = f"{host}:{port}" if port else host
    return f"{proto}//{netloc}{path}{q}"


def normalize_url(url: str, base: Optional[str] = None) -> Optional[str]:
    try:
        u = ada_url.URL(url, base) if base else ada_url.URL(url)
    except Exception:
        return None
    return _finish(u)


def extract(html: bytes, base_url: str, max_text: int, max_links: int):
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    lang = ""
    h = tree.css_first("html")
    if h is not None:
        lang = (h.attributes.get("lang") or "")[:8]
    links: list[str] = []
    seen: set[str] = set()
    for a in tree.css("a[href]"):
        href = a.attributes.get("href")
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
            continue
        u = normalize_url(href, base_url)
        if u and u not in seen:
            seen.add(u)
            links.append(u)
            if len(links) >= max_links:
                break
    tree.strip_tags(STRIP_TAGS)
    body = tree.body
    text = body.text(separator="\n", strip=True) if body is not None else ""
    text = _ws.sub(" ", text)
    text = _nl.sub("\n", text)
    if len(text) > max_text:
        text = text[:max_text]
    return title, lang, text, links


# ---------------------------------------------------------------- DNS
class DnsPyResolver(aiohttp.abc.AbstractResolver):
    """Pure-asyncio DNS (dnspython) with positive/negative caching and a concurrency cap."""

    def __init__(self, nameservers: list[str], concurrency: int = 256, ttl: float = 900, neg_ttl: float = 600):
        import dns.asyncresolver
        self.r = dns.asyncresolver.Resolver(configure=False)
        self.r.nameservers = nameservers
        self.r.lifetime = 4.0
        self.r.timeout = 2.0
        self.r.rotate = True
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: dict[str, tuple[float, list[str]]] = {}
        self.ttl, self.neg_ttl = ttl, neg_ttl
        self.stats = collections.Counter()

    async def resolve(self, host: str, port: int = 0, family=socket.AF_INET):
        now = time.time()
        c = self.cache.get(host)
        if c and c[0] > now:
            addrs = c[1]
            self.stats["hit"] += 1
        else:
            try:
                async with self.sem:
                    ans = await self.r.resolve(host, "A", search=False)
                addrs = [r.address for r in ans]
                self.stats["miss"] += 1
            except Exception as e:
                addrs = []
                self.stats["fail"] += 1
            if len(self.cache) > 500_000:
                self.cache.clear()
            self.cache[host] = (now + (self.ttl if addrs else self.neg_ttl), addrs)
        if not addrs:
            raise OSError(f"DNS lookup failed: {host}")
        return [{"hostname": host, "host": a, "port": port, "family": socket.AF_INET, "proto": 0,
                 "flags": socket.AI_NUMERICHOST} for a in addrs]

    async def close(self):
        pass


def system_nameservers() -> list[str]:
    try:
        import dns.resolver
        ns = [n for n in dns.resolver.Resolver().nameservers if ":" not in n]
    except Exception:
        ns = []
    return ns or ["1.1.1.1", "8.8.8.8"]


# ---------------------------------------------------------------- frontier
@dataclass(order=True)
class HostEntry:
    ready_at: float
    host: str = field(compare=False)


@dataclass
class Host:
    name: str
    queue: collections.deque = field(default_factory=collections.deque)
    inflight: int = 0
    done: int = 0
    errors: int = 0
    robots: Optional[Protego] = None
    robots_state: int = 0  # 0 unknown, 1 fetching, 2 ready
    waiters: list = field(default_factory=list)
    crawl_delay: float = 0.0
    scheduled: bool = False


class Stats:
    def __init__(self):
        self.t0 = time.time()
        self.pages = 0
        self.bytes = 0
        self.wire_bytes = 0
        self.errors = collections.Counter()
        self.status = collections.Counter()
        self.lat = collections.defaultdict(list)

    def add_lat(self, stage, ms):
        l = self.lat[stage]
        l.append(ms)
        if len(l) > 20000:
            del l[: len(l) // 2]

    def snapshot(self, final=False):
        return {"pages": self.pages, "bytes": self.bytes, "wire_bytes": self.wire_bytes, "errors": dict(self.errors),
                "status": {str(k): v for k, v in self.status.items()},
                "lat": {k: (v if final else v[-300:]) for k, v in self.lat.items()}}


def pct(samples, p):
    if not samples:
        return float("nan")
    s = sorted(samples)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


class Crawler:
    def __init__(self, out_path: str, cfg: dict, wid: int = 0, nw: int = 1, inboxes=None, stats_q=None):
        self.out_path = out_path
        self.cfg = cfg
        self.concurrency = cfg["concurrency"]
        self.per_host = cfg["per_host"]
        self.host_delay = cfg["host_delay"]
        self.max_per_host = cfg["max_per_host"]
        self.time_limit = cfg["time_limit"]
        self.max_pages = cfg["max_pages"]
        self.max_text = cfg.get("max_text", 20000)
        self.max_links = cfg.get("max_links", 150)
        self.max_body = cfg.get("max_body", 600_000)
        self.max_depth = cfg["max_depth"]
        self.log_every = cfg.get("log_every", 5.0)
        self.shard_pages = cfg.get("shard_pages", 20000)
        self.frontier_cap = cfg.get("frontier_cap", 3_000_000)
        self.wid, self.nw = wid, nw
        self.inboxes = inboxes
        self.stats_q = stats_q
        self.outbox: list[list] = [[] for _ in range(nw)]
        self.outbox_n = 0

        self.hosts: dict[str, Host] = {}
        self.ready: list[HostEntry] = []
        self.seen: set[int] = set()
        self.frontier_size = 0
        self.stats = Stats()
        self.stop = False
        self.inflight = 0
        self.records_in_shard = 0
        self.shard_idx = 0
        self._writer = None
        self._fh = None
        self.wake: asyncio.Event
        self.resolver: Optional[DnsPyResolver] = None

    # ---------- output ----------
    def _open_shard(self):
        if self._writer:
            self._writer.close(); self._fh.close()
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        path = f"{self.out_path}.w{self.wid:02d}.{self.shard_idx:04d}.jsonl.zst"
        self._fh = open(path, "wb")
        self._writer = zstd.ZstdCompressor(level=3).stream_writer(self._fh)
        self.shard_idx += 1
        self.records_in_shard = 0

    def write(self, rec: dict):
        if self._writer is None or self.records_in_shard >= self.shard_pages:
            self._open_shard()
        self._writer.write(orjson.dumps(rec) + b"\n")
        self.records_in_shard += 1

    def close(self):
        if self._writer:
            self._writer.close(); self._fh.close(); self._writer = None

    # ---------- frontier ----------
    def owner(self, key: str) -> int:
        return xxhash.xxh32_intdigest(key.encode()) % self.nw if self.nw > 1 else 0

    def add_url(self, url: str, depth: int, local_only: bool = False):
        if depth > self.max_depth:
            return
        hu = xxhash.xxh64_intdigest(url.encode())
        if hu in self.seen:
            return
        try:
            host = ada_url.URL(url).hostname
        except Exception:
            return
        if not host:
            return
        key = site_key(host)
        if not local_only and self.nw > 1:
            o = self.owner(key)
            if o != self.wid:
                self.seen.add(hu)  # don't route the same url twice from this worker
                self.outbox[o].append((url, depth))
                self.outbox_n += 1
                return
        if self.frontier_size >= self.frontier_cap:
            self.stats.errors["frontier_full"] += 1
            return
        h = self.hosts.get(key)
        if h is None:
            h = self.hosts[key] = Host(key)
        if h.done + len(h.queue) >= self.max_per_host:
            return
        self.seen.add(hu)
        h.queue.append((url, depth))
        self.frontier_size += 1
        self._schedule(h, time.time())

    def flush_outbox(self, force=False):
        if self.nw <= 1 or (self.outbox_n == 0):
            return
        for o, batch in enumerate(self.outbox):
            if batch and (force or len(batch) >= 200 or True):
                try:
                    self.inboxes[o].put_nowait(batch)
                except Exception:
                    self.stats.errors["route_fail"] += 1
                self.outbox[o] = []
        self.outbox_n = 0

    def drain_inbox(self, max_batches=200):
        if self.nw <= 1:
            return 0
        q = self.inboxes[self.wid]
        n = 0
        for _ in range(max_batches):
            try:
                batch = q.get_nowait()
            except Exception:
                break
            for url, depth in batch:
                self.add_url(url, depth, local_only=True)
                n += 1
        return n

    def _schedule(self, h: Host, when: float):
        if not h.scheduled and h.queue and h.inflight < self.per_host:
            h.scheduled = True
            heapq.heappush(self.ready, HostEntry(when, h.name))
            self.wake.set()

    def next_url(self):
        now = time.time()
        while self.ready:
            e = self.ready[0]
            if e.ready_at > now:
                return None, e.ready_at - now
            heapq.heappop(self.ready)
            h = self.hosts[e.host]
            h.scheduled = False
            if not h.queue or h.inflight >= self.per_host:
                continue
            url, depth = h.queue.popleft()
            self.frontier_size -= 1
            h.inflight += 1
            self._schedule(h, now + self.host_delay)
            return (url, depth, h), 0
        return None, None

    # ---------- robots ----------
    async def robots_allowed(self, session, h: Host, url: str) -> bool:
        if h.robots_state == 2:
            return h.robots is None or h.robots.can_fetch(url, USER_AGENT)
        if h.robots_state == 1:
            fut = asyncio.get_running_loop().create_future()
            h.waiters.append(fut)
            await fut
            return h.robots is None or h.robots.can_fetch(url, USER_AGENT)
        h.robots_state = 1
        p = urlsplit(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        t = time.perf_counter()
        try:
            async with session.get(robots_url, allow_redirects=True) as r:
                if r.status == 200 and (r.content_type or "").startswith("text"):
                    body = await r.content.read(200_000)
                    rp = Protego.parse(body.decode("utf-8", "replace"))
                    h.robots = rp
                    cd = rp.crawl_delay(USER_AGENT)
                    if cd:
                        h.crawl_delay = min(float(cd), 10.0)
                elif r.status in (401, 403):
                    h.robots = Protego.parse("User-agent: *\nDisallow: /\n")
                else:
                    h.robots = None
        except Exception:
            h.robots = None
        self.stats.add_lat("robots", (time.perf_counter() - t) * 1000)
        h.robots_state = 2
        for fut in h.waiters:
            if not fut.done():
                fut.set_result(True)
        h.waiters.clear()
        return h.robots is None or h.robots.can_fetch(url, USER_AGENT)

    # ---------- fetch ----------
    async def fetch(self, session, url: str, depth: int, h: Host):
        st = self.stats
        ctx: dict = {}
        t0 = time.perf_counter()
        try:
            if not await self.robots_allowed(session, h, url):
                st.errors["robots_disallow"] += 1
                return
            if h.crawl_delay > self.host_delay:
                await asyncio.sleep(h.crawl_delay - self.host_delay)
            async with session.get(url, allow_redirects=True, max_redirects=5, trace_request_ctx=ctx) as r:
                st.status[r.status] += 1
                if r.status != 200:
                    st.errors[f"http_{r.status//100}xx"] += 1
                    return
                ct = r.headers.get("Content-Type", "")
                if "html" not in ct:
                    st.errors["not_html"] += 1
                    return
                t_ttfb = time.perf_counter()
                body = await r.content.read(self.max_body)
                truncated = not r.content.at_eof()
                if truncated:
                    st.errors["truncated"] += 1
                    r.close()  # don't reuse connection with unread body
                t_dl = time.perf_counter()
                final_url = str(r.url)
                cl = r.headers.get("Content-Length")
                wire = int(cl) if cl and cl.isdigit() else len(body) // 4
            title, lang, text, links = extract(body, final_url, self.max_text, self.max_links)
            t_parse = time.perf_counter()
            if len(text) < 200:
                st.errors["too_short"] += 1
            else:
                timings = {"dns": ctx.get("dns", 0.0), "connect": ctx.get("connect", 0.0),
                           "ttfb": (t_ttfb - t0) * 1000, "download": (t_dl - t_ttfb) * 1000,
                           "parse": (t_parse - t_dl) * 1000, "total": (t_parse - t0) * 1000}
                for k, v in timings.items():
                    st.add_lat(k, v)
                st.pages += 1
                st.bytes += len(body)
                st.wire_bytes += wire
                self.write({"url": final_url, "host": h.name, "title": title, "lang": lang, "text": text,
                            "depth": depth, "n_links": len(links), "bytes": len(body), "ts": time.time(), "t": timings})
            h.done += 1
            if depth < self.max_depth:
                for l in links:
                    self.add_url(l, depth + 1)
        except asyncio.TimeoutError:
            st.errors["timeout"] += 1; h.errors += 1
        except aiohttp.ClientError as e:
            st.errors[type(e).__name__] += 1; h.errors += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            st.errors["exc_" + type(e).__name__] += 1
        finally:
            h.inflight -= 1
            self.inflight -= 1
            self._schedule(h, time.time() + self.host_delay)
            self.wake.set()

    # ---------- main loop ----------
    async def run(self, seeds: list[str]):
        self.wake = asyncio.Event()
        for s in seeds:
            u = normalize_url(s)
            if u:
                self.add_url(u, 0, local_only=True)
        trace = aiohttp.TraceConfig()

        async def on_dns_start(sess, ctx, params):
            ctx.trace_request_ctx["_dns0"] = time.perf_counter()

        async def on_dns_end(sess, ctx, params):
            c = ctx.trace_request_ctx
            if "_dns0" in c:
                c["dns"] = (time.perf_counter() - c["_dns0"]) * 1000

        async def on_conn_start(sess, ctx, params):
            ctx.trace_request_ctx["_c0"] = time.perf_counter()

        async def on_conn_end(sess, ctx, params):
            c = ctx.trace_request_ctx
            if "_c0" in c:
                c["connect"] = (time.perf_counter() - c["_c0"]) * 1000

        trace.on_dns_resolvehost_start.append(on_dns_start)
        trace.on_dns_resolvehost_end.append(on_dns_end)
        trace.on_connection_create_start.append(on_conn_start)
        trace.on_connection_create_end.append(on_conn_end)

        self.resolver = DnsPyResolver(self.cfg["nameservers"], concurrency=self.cfg.get("dns_concurrency", 256))
        connector = aiohttp.TCPConnector(limit=self.concurrency + 64, limit_per_host=self.per_host, use_dns_cache=False,
                                         resolver=self.resolver, enable_cleanup_closed=True, happy_eyeballs_delay=None)
        timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=8)
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
                   "Accept-Language": "en;q=0.9,*;q=0.5", "Accept-Encoding": "gzip, deflate, br"}
        tasks: set[asyncio.Task] = set()
        t_start = time.time()
        last_log = t_start
        last_flush = t_start
        idle_since = None
        session = aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers, trace_configs=[trace],
                                        auto_decompress=True, read_bufsize=256 * 1024)
        try:
            while not self.stop:
                now = time.time()
                if self.stats.pages >= self.max_pages or (self.time_limit and now - t_start > self.time_limit):
                    break
                if now - last_flush >= 0.25:
                    self.flush_outbox(force=True)
                    self.drain_inbox()
                    last_flush = now
                if now - last_log >= self.log_every:
                    self.report(False)
                    last_log = now
                launched = 0
                while self.inflight < self.concurrency:
                    item, wait = self.next_url()
                    if item is None:
                        break
                    url, depth, h = item
                    self.inflight += 1
                    t = asyncio.create_task(self.fetch(session, url, depth, h))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
                    launched += 1
                if launched == 0:
                    if self.inflight == 0 and not self.ready:
                        if self.nw <= 1:
                            break  # frontier exhausted
                        idle_since = idle_since or now
                        if now - idle_since > 20:
                            break
                    else:
                        idle_since = None
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), timeout=min(wait, 0.25) if wait else 0.25)
                    except asyncio.TimeoutError:
                        pass
            self.stop = True
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await asyncio.wait_for(session.close(), timeout=5)
            except Exception:
                pass
            self.close()
        elapsed = time.time() - t_start
        self.report(True, elapsed)
        return self.summary(elapsed)

    def report(self, final: bool, elapsed: float = 0.0):
        snap = self.stats.snapshot(final)
        snap.update({"wid": self.wid, "final": final, "frontier": self.frontier_size, "hosts": len(self.hosts),
                     "inflight": self.inflight, "seen": len(self.seen), "elapsed": elapsed,
                     "dns": dict(self.resolver.stats) if self.resolver else {}})
        if self.stats_q is not None:
            self.stats_q.put(snap)
        else:
            print(format_line(time.time() - self.stats.t0, [snap], self.stats.t0), flush=True)

    def summary(self, elapsed):
        st = self.stats
        return {"elapsed_s": elapsed, "pages": st.pages, "pages_per_s": st.pages / max(elapsed, 1e-6),
                "mbytes": st.bytes / 1e6, "errors": dict(st.errors), "hosts": len(self.hosts), "seen_urls": len(self.seen),
                "latency_ms": {k: {"p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99)} for k, v in st.lat.items()}}


_prev = {"pages": 0, "t": time.time()}


def format_line(elapsed, snaps, t0):
    pages = sum(s["pages"] for s in snaps)
    mb = sum(s["bytes"] for s in snaps) / 1e6
    wire = sum(s["wire_bytes"] for s in snaps) / 1e6
    err = sum(sum(s["errors"].values()) for s in snaps)
    frontier = sum(s["frontier"] for s in snaps)
    hosts = sum(s["hosts"] for s in snaps)
    inflight = sum(s["inflight"] for s in snaps)
    now = time.time()
    inst = (pages - _prev["pages"]) / max(1e-6, now - _prev["t"])
    _prev.update(pages=pages, t=now)
    lat = collections.defaultdict(list)
    for s in snaps:
        for k, v in s["lat"].items():
            lat[k].extend(v)
    return (f"[{elapsed:6.0f}s] pages={pages} ({pages/max(elapsed,1e-6):.1f}/s avg, {inst:.1f}/s now) MB={mb:.0f} "
            f"wireMB={wire:.0f} err={err} frontier={frontier} hosts={hosts} inflight={inflight} | p50 dns={pct(lat['dns'],50):.0f} "
            f"conn={pct(lat['connect'],50):.0f} ttfb={pct(lat['ttfb'],50):.0f} dl={pct(lat['download'],50):.0f} "
            f"parse={pct(lat['parse'],50):.1f} total={pct(lat['total'],50):.0f}ms p95 total={pct(lat['total'],95):.0f}ms")


# ---------------------------------------------------------------- multiprocess
def _worker_main(wid, nw, seeds, inboxes, stats_q, out, cfg):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    c = Crawler(out, cfg, wid, nw, inboxes, stats_q)
    try:
        asyncio.run(c.run(seeds))
    except Exception as e:  # pragma: no cover
        stats_q.put({"wid": wid, "final": True, "crash": repr(e), "pages": 0, "bytes": 0, "wire_bytes": 0, "errors": {},
                     "status": {}, "lat": {}, "frontier": 0, "hosts": 0, "inflight": 0, "seen": 0, "elapsed": 0, "dns": {}})


def run_multi(seeds, out, cfg, nw):
    ctx = mp.get_context("spawn")
    inboxes = [ctx.Queue(maxsize=30000) for _ in range(nw)]
    stats_q = ctx.Queue()
    per = [[] for _ in range(nw)]
    for s in seeds:
        u = normalize_url(s)
        if u:
            per[xxhash.xxh32_intdigest(site_key(ada_url.URL(u).hostname).encode()) % nw].append(u)
    procs = [ctx.Process(target=_worker_main, args=(w, nw, per[w], inboxes, stats_q, out, cfg), daemon=True) for w in range(nw)]
    t0 = time.time()
    for p in procs:
        p.start()
    latest = {}
    finals = {}
    last_print = t0
    while len(finals) < nw:
        try:
            s = stats_q.get(timeout=1.0)
        except Exception:
            s = None
        if s is not None:
            if s.get("final"):
                finals[s["wid"]] = s
            else:
                latest[s["wid"]] = s
        now = time.time()
        if now - last_print >= cfg.get("log_every", 5.0) and latest:
            print(format_line(now - t0, list(latest.values()), t0), flush=True)
            last_print = now
        if cfg["time_limit"] and now - t0 > cfg["time_limit"] + 60:
            print("workers slow to exit; terminating", flush=True)
            break
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    elapsed = time.time() - t0
    snaps = list(finals.values()) or list(latest.values())
    print(format_line(elapsed, snaps, t0), flush=True)
    errors = collections.Counter(); status = collections.Counter(); lat = collections.defaultdict(list); dns = collections.Counter()
    for s in snaps:
        errors.update(s["errors"]); status.update(s["status"]); dns.update(s.get("dns", {}))
        for k, v in s["lat"].items():
            lat[k].extend(v)
    pages = sum(s["pages"] for s in snaps)
    return {"elapsed_s": elapsed, "pages": pages, "pages_per_s": pages / elapsed, "mbytes": sum(s["bytes"] for s in snaps) / 1e6,
            "wire_mbytes": sum(s["wire_bytes"] for s in snaps) / 1e6, "errors": dict(errors), "status": dict(status), "dns": dict(dns),
            "hosts": sum(s["hosts"] for s in snaps), "seen_urls": sum(s["seen"] for s in snaps),
            "crashes": [s["crash"] for s in snaps if s.get("crash")],
            "latency_ms": {k: {"p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99)} for k, v in lat.items()},
            "config": {**cfg, "workers": nw}}


def load_seeds(path: str, n: int, offset: int = 0) -> list[str]:
    seeds = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(seeds) >= n:
                break
            line = line.strip()
            if not line:
                continue
            dom = line.split(",")[-1] if "," in line else line
            if "://" not in dom:
                dom = "https://" + dom
            seeds.append(dom)
    return seeds


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="data/seeds/top-1m.csv")
    ap.add_argument("--n-seeds", type=int, default=2000)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", default="data/crawl/pages")
    ap.add_argument("--max-pages", type=int, default=10000, help="per worker")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=512, help="per worker")
    ap.add_argument("--per-host", type=int, default=2)
    ap.add_argument("--host-delay", type=float, default=0.5)
    ap.add_argument("--max-per-host", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-body", type=int, default=600_000)
    ap.add_argument("--max-links", type=int, default=150)
    ap.add_argument("--time-limit", type=float, default=0)
    ap.add_argument("--nameservers", default="", help="comma list; default system + public")
    ap.add_argument("--summary", default="")
    args = ap.parse_args(argv)

    ns = [x for x in args.nameservers.split(",") if x] or (system_nameservers() + ["1.1.1.1", "8.8.8.8"])
    cfg = {"concurrency": args.concurrency, "per_host": args.per_host, "host_delay": args.host_delay,
           "max_per_host": args.max_per_host, "max_depth": args.max_depth, "time_limit": args.time_limit,
           "max_pages": args.max_pages, "max_body": args.max_body, "max_links": args.max_links, "nameservers": ns}
    seeds = load_seeds(args.seeds, args.n_seeds, args.seed_offset)
    if args.workers <= 1:
        if uvloop is not None:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        c = Crawler(args.out, cfg)

        def _sig(*_):
            c.stop = True
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        summary = asyncio.run(c.run(seeds))
        summary["config"] = {**cfg, "workers": 1}
    else:
        summary = run_multi(seeds, args.out, cfg, args.workers)
    print(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or ".", exist_ok=True)
        with open(args.summary, "wb") as f:
            f.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
