"""Live index: the prebuilt HNSW index plus passages added at runtime, searchable the moment they arrive.

Tiered like an LSM tree, so writes never block reads:
  base HNSW (immutable, 2.4M)  +  sealed HNSW segments (50k each)  +  flat fresh tier (< 50k, exact matvec)
A query searches all tiers and merges by score. New vectors are appended to the fresh tier (O(1), no lock on the
read path); every 50k the oldest rows are sealed into an HNSW segment built by a separate faiss-only process
(full OpenMP there; in this process OpenMP must stay single-threaded because torch's runtime is loaded too).
Every 4 adjacent equal-size segments are compacted into one, so the segment count (each costs ~0.2 ms per query)
grows with log(live passages) rather than linearly. Segments are cached by content hash, so restarts reuse them.

Where vectors come from:
  - live crawl: crawler -> embed_watch (separate processes) -> parts in data/index/parts_live/, tailed here
  - URLs pasted in chat: fetched and embedded in-process on the GPU (latency path, the LLM is idle at that point)
Runtime additions persist (parts files, data/index/live/delta.*) and are reloaded on start.

Placement on Apple silicon: query embeddings on the ANE (32-token graph, 0.7 ms), HNSW + flat search on the CPU,
passage embeddings on the GPU when the LLM is idle (1,145 passages/s at 256 tokens) or the ANE (258/s) when not.
"""
from __future__ import annotations

import asyncio
import glob
import io
import os
import subprocess
import sys
import threading
import time

import numpy as np
import orjson
import zstandard as zstd

from .corpus import chunk_text
from .crawler import USER_AGENT, extract, normalize_url
from .query import Engine

MODEL = "BAAI/bge-small-en-v1.5"
PARTS_LIVE = "data/index/parts_live"


class _Docs:
    """Base doc-id array plus appended ids, indexable like the array Engine.search expects."""

    def __init__(self, base: np.ndarray):
        self.base, self.n, self.extra = base, len(base), []

    def __getitem__(self, i):
        return self.base[i] if i < self.n else self.extra[i - self.n]


class _Passages:
    """Base mmap passage store plus appended passage texts."""

    def __init__(self, base, n: int):
        self.base, self.n, self.extra = base, n, []

    def get(self, i: int, max_chars: int = 300) -> str:
        return self.base.get(i, max_chars) if i < self.n else self.extra[i - self.n][:max_chars]


class Tiered:
    """Searcher over base HNSW + sealed segments + flat fresh tier. Readers take one snapshot of `state`
    (a tuple swapped atomically), so appends and seals never block or tear a search."""

    def __init__(self, base, base_n: int, dim: int, seg_size: int = 50_000, seg_dir: str = "data/index/live/segments"):
        self.base, self.dim, self.seg_size, self.seg_dir = base, dim, seg_size, seg_dir
        # (fresh buffer, rows used, pid of row 0, ((pid0, size, segment index, vectors path), ...))
        self.state = (np.empty((8192, dim), np.float32), 0, base_n, ())
        self.wlock = threading.Lock()
        self.sealing = False
        os.makedirs(seg_dir, exist_ok=True)

    def set_param(self, ef):
        self.base.set_param(ef)

    def append(self, vecs: np.ndarray):
        with self.wlock:
            buf, n, start, segs = self.state
            need = n + len(vecs)
            if need > len(buf):
                nb = np.empty((max(need, 2 * len(buf)), self.dim), np.float32); nb[:n] = buf[:n]; buf = nb
            buf[n:need] = vecs  # rows past n are invisible to readers until state is swapped
            self.state = (buf, need, start, segs)
            if need >= self.seg_size and not self.sealing:
                self.sealing = True
                threading.Thread(target=self._seal, daemon=True, name="seal").start()

    def _build(self, rows: np.ndarray):
        """HNSW segment for these vectors, from cache or built by a faiss-only process (every core)."""
        import faiss
        import xxhash
        base = os.path.join(self.seg_dir, f"seg_{xxhash.xxh3_64_hexdigest(rows.tobytes())}")
        if not os.path.exists(base + ".hnsw"):
            np.save(base + ".npy", rows)  # kept: compaction rebuilds bigger segments from these
            env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
            subprocess.run([sys.executable, "-m", "search.live", "seal", base + ".npy", base + ".hnsw"], env=env, check=True)
        elif not os.path.exists(base + ".npy"):
            np.save(base + ".npy", rows)
        idx = faiss.read_index(base + ".hnsw")
        idx.hnsw.efSearch = 64
        return idx, base + ".npy"

    def _seal(self):
        try:
            while True:
                buf, n, start, segs = self.state
                if n < self.seg_size:
                    break
                idx, vpath = self._build(buf[:self.seg_size].copy())
                with self.wlock:
                    buf, n, start, segs = self.state
                    rest = np.empty((max(8192, 2 * (n - self.seg_size)), self.dim), np.float32)
                    rest[:n - self.seg_size] = buf[self.seg_size:n]
                    self.state = (rest, n - self.seg_size, start + self.seg_size, segs + ((start, self.seg_size, idx, vpath),))
            self._compact()
            # everything is sealed now, so any segment file not in use is a leftover of an older layout
            used = {v[:-4] for *_, v in self.state[3]}
            for f in glob.glob(os.path.join(self.seg_dir, "seg_*")):
                if f.rsplit(".", 1)[0] not in used:
                    os.remove(f)
        finally:
            with self.wlock:
                self.sealing = False
                # appends that crossed seg_size while this thread was finishing saw sealing=True and started nothing
                if self.state[1] >= self.seg_size:
                    self.sealing = True
                    threading.Thread(target=self._seal, daemon=True, name="seal").start()

    def _compact(self, fan_in: int = 4):
        """Merge the last `fan_in` segments while they are equal-sized (size-tiered, like an LSM tree)."""
        while True:
            segs = self.state[3]
            tail = segs[-fan_in:]
            if len(tail) < fan_in or len({sz for _, sz, _, _ in tail}) != 1:
                return
            rows = np.concatenate([np.load(v) for _, _, _, v in tail])
            idx, vpath = self._build(rows)
            with self.wlock:
                buf, n, start, segs = self.state
                self.state = (buf, n, start, segs[:-fan_in] + ((tail[0][0], len(rows), idx, vpath),))
            for _, _, _, v in tail:  # the merged segment's vectors file now covers these rows
                for f in (v, v[:-4] + ".hnsw"):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    def search(self, q: np.ndarray, k: int):
        buf, n, start, segs = self.state
        D, I = self.base.search(q, k)
        if not n and not segs:
            return D, I
        ds, ids = [D[0]], [I[0]]
        for pid0, _, idx, _ in segs:
            d, i = idx.search(q, k)
            ds.append(d[0]); ids.append(np.where(i[0] >= 0, i[0] + pid0, -1))
        if n:
            s = buf[:n] @ q[0]
            kk = min(k, n)
            top = np.argpartition(-s, kk - 1)[:kk]
            ds.append(s[top]); ids.append(top + start)
        d, i = np.concatenate(ds), np.concatenate(ids)
        o = np.argsort(-d)[:k]
        return d[None, o], i[None, o]


class LiveIndex:
    def __init__(self, emb_prefix="data/index/emb_all", index_path="data/index/idx_all.hnsw", live_dir="data/index/live",
                 ef: int = 64, ingest_batch: int = 16, gpu_busy: threading.Event | None = None):
        t0 = time.perf_counter()
        from .embed import make_embedder
        # Passage encoders first: loading a CoreML model after faiss is in the process segfaults
        self.B = ingest_batch
        self.ingest_gpu = make_embedder("coreml", MODEL, max_seq=256, batch_size=ingest_batch)
        self.ingest_ane = make_embedder("coreml_ane", MODEL, max_seq=256, batch_size=ingest_batch)
        self.gpu_busy = gpu_busy or threading.Event()  # set by the LLM while it generates
        # 32-token batch-1 query graph on the ANE: 0.72 ms p50 vs 2.4 ms for the 64-token GPU graph
        self.eng = Engine(MODEL, emb_prefix, index_path, "hnsw", backend="coreml_ane", ef=ef, max_seq=32)
        self.base_n = len(self.eng.urls)
        assert self.eng.searcher.idx.ntotal == self.base_n, "index and metadata disagree"
        from .wiki import WikiIndex
        from .wikidata import Cards
        self.wiki = WikiIndex()  # Wikipedia tier (grows while `search.wiki embed/index` run; reloaded by wiki_reload)
        self.cards = Cards()     # Wikidata fact cards + exact-name entity lookup
        self.tiers = Tiered(self.eng.searcher, self.base_n, self.eng.searcher.idx.d)
        self.eng.searcher = self.tiers
        self.eng.docs = _Docs(self.eng.docs)
        self.eng.passages = _Passages(self.eng.passages, self.base_n)
        self.next_doc = int(self.eng.docs.base.max()) + 1
        self.seen = set(self.eng.urls)
        self.wlock = threading.Lock()  # one writer at a time keeps pid order == metadata order

        self.ane_free = threading.Event(); self.ane_free.set()
        self.last_query = time.time()
        for q in ("warm up the neural engine", "a second warmup query", "and a third one"):
            self.eng.emb.encode([q], is_query=True)
        for emb in (self.ingest_gpu, self.ingest_ane):  # first predict on each graph compiles it (~150 ms)
            emb.encode(["warm up passage encoder"] * ingest_batch)
        # snippet reads are random 1 KB reads into a 3 GB file: pull it into the page cache up front
        threading.Thread(target=self._pretouch, args=(emb_prefix + ".passages.bin",), daemon=True).start()

        self.live_dir = live_dir
        os.makedirs(live_dir, exist_ok=True)
        self.parts_loaded: set[str] = set()
        self.stats = {"live_passages": 0, "live_pages": 0, "ingest_pps": 0.0, "last_ingest": None, "ingest_unit": None}
        self._load_delta()
        self.load_parts()
        self.t_load = time.perf_counter() - t0

    # ------------------------------------------------------------------ query
    def search(self, query: str, k: int = 8) -> dict:
        self.ane_free.clear()
        try:
            t0 = time.perf_counter()
            qv = self.eng.embed(query)  # per request: concurrent requests must not share a query vector
            t_embed = (time.perf_counter() - t0) * 1e3
            res = self.eng.search(query, k, qv=qv)
            res["t_ms"]["embed"] = t_embed; res["t_ms"]["total"] += t_embed
            t0 = time.perf_counter()
            res["wiki"] = self.wiki.search(qv, k)
            res["t_ms"]["wiki"] = (time.perf_counter() - t0) * 1e3
            res["t_ms"]["total"] += res["t_ms"]["wiki"]
            return res
        finally:
            self.last_query = time.time()
            self.ane_free.set()

    def wiki_reload(self, every: float = 60.0):
        """Pick up Wikipedia passages indexed since the last check, and the Wikidata DB once it has been built."""
        from .wikidata import Cards
        while True:
            time.sleep(every)
            try:
                if self.wiki.reload():
                    print(f"[wiki] {self.wiki.n:,} passages", flush=True)
                if not self.cards.ok:
                    c = Cards()
                    if c.ok:
                        self.cards = c; print("[wikidata] cards loaded", flush=True)
            except Exception as e:  # a half-written index between runs: try again next round
                print(f"[wiki] reload failed: {e!r}", flush=True)

    def keep_warm(self, every: float = 3.0):
        """The ANE powers down after a few idle seconds and the next query embed costs 23-29 ms instead of ~1 ms.
        One 0.7 ms inference every few idle seconds keeps it awake."""
        while True:
            time.sleep(every)
            if time.time() - self.last_query >= every:
                self.eng.emb.encode(["keep warm"], is_query=True)

    def embed_query(self, query: str) -> np.ndarray:
        self.ane_free.clear()
        try:
            return self.eng.emb.encode([query], is_query=True)[0]
        finally:
            self.ane_free.set()

    def passage(self, pid: int, max_chars: int) -> str:
        return self.eng.passages.get(pid, max_chars)

    @property
    def size(self) -> int:
        return len(self.eng.urls)

    def status(self) -> dict:
        buf, n, start, segs = self.tiers.state
        return {**self.stats, "segments": len(segs), "fresh": n, "wiki_passages": self.wiki.n, "wikidata": self.cards.ok}

    @staticmethod
    def _pretouch(path: str, block: int = 8 << 20):
        try:
            with open(path, "rb", buffering=0) as f:
                while f.read(block):
                    pass
        except OSError:
            pass

    # ------------------------------------------------------------------ writes
    def _append(self, vecs, metas, texts, persist_src=None):
        """metas: [(url, title, doc_id)]. Caller holds self.wlock. Metadata goes first so a reader that sees the
        new vectors can always resolve their pids."""
        for (u, t, d), tx in zip(metas, texts):
            self.eng.urls.append(u); self.eng.titles.append(t)
            self.eng.docs.extra.append(d); self.eng.passages.extra.append(tx)
        self.tiers.append(np.ascontiguousarray(vecs, dtype=np.float32))
        self.stats["live_passages"] += len(texts)
        if persist_src is not None:
            with open(os.path.join(self.live_dir, "delta.f16"), "ab") as f:
                f.write(vecs.astype(np.float16).tobytes())
            with open(os.path.join(self.live_dir, "delta.jsonl"), "ab") as f:
                for (u, t, d), tx in zip(metas, texts):
                    f.write(orjson.dumps({"u": u, "t": t, "d": d, "x": tx, "s": persist_src}) + b"\n")

    def add_pages(self, pages: list[dict], src: str, max_chunks: int = 6, rank_all: bool = False) -> list[dict]:
        """Chunk -> embed -> append new pages. With rank_all, pages already indexed are embedded too (fresh text,
        not re-added). Returns [{url, title, texts, vecs, new}] so callers can rank a page's chunks directly."""
        items = []  # (url, title, is_new, passages)
        with self.wlock:
            for p in pages:
                url = p["url"]
                new = url not in self.seen
                if not p.get("text") or not (new or rank_all):
                    continue
                chunks = chunk_text(p["text"], 180, 30, max_chunks)
                if chunks:
                    if new:
                        self.seen.add(url)  # claim now so a concurrent fetch of the same page doesn't double-add
                    title = (p.get("title") or "").strip()[:200]
                    items.append((url, title, new, [f"{title}\n{ch}" if title else ch for ch in chunks]))
        texts = [t for it in items for t in it[3]]
        if not texts:
            return []
        t0 = time.perf_counter()
        parts = []
        for i in range(0, len(texts), self.B):
            if self.gpu_busy.is_set():
                self.ane_free.wait()  # a query is embedding: let it go first
                emb, self.stats["ingest_unit"] = self.ingest_ane, "ane"
            else:
                emb, self.stats["ingest_unit"] = self.ingest_gpu, "gpu"
            parts.append(emb.encode(texts[i:i + self.B]).astype(np.float32))
        vecs = np.concatenate(parts)
        out, add_v, add_m, add_t, j = [], [], [], [], 0
        with self.wlock:
            for url, title, new, ts in items:
                v = vecs[j:j + len(ts)]; j += len(ts)
                if new:
                    d = self.next_doc; self.next_doc += 1
                    add_v.append(v); add_m += [(url, title, d)] * len(ts); add_t += ts
                out.append({"url": url, "title": title, "texts": ts, "vecs": v, "new": new})
            if add_v:
                self._append(np.concatenate(add_v), add_m, add_t, persist_src=src)
        self.stats["ingest_pps"] = round(len(texts) / (time.perf_counter() - t0), 1)
        self.stats["live_pages"] += sum(o["new"] for o in out)
        self.stats["last_ingest"] = time.time()
        return out

    def _load_delta(self):
        vp, mp = os.path.join(self.live_dir, "delta.f16"), os.path.join(self.live_dir, "delta.jsonl")
        if not (os.path.exists(vp) and os.path.exists(mp)):
            return
        rows, ends = [], []  # ends[i]: byte offset just past row i's line
        with open(mp, "rb") as f:
            for line in f:
                if not line.endswith(b"\n"):
                    break  # a crash mid-write left a torn last line
                rows.append(orjson.loads(line)); ends.append((ends[-1] if ends else 0) + len(line))
        dim = self.tiers.dim
        vecs = np.fromfile(vp, dtype=np.float16)
        n = min(len(rows), len(vecs) // dim)  # a crash between the two writes leaves at most one batch unmatched
        # cut both files back to the n matched rows, or every later append would pair row i with another vector
        if len(vecs) != n * dim:
            os.truncate(vp, n * dim * 2)
        if os.path.getsize(mp) != (ends[n - 1] if n else 0):
            os.truncate(mp, ends[n - 1] if n else 0)
        if not n:
            return
        rows = rows[:n]
        with self.wlock:
            self._append(vecs[:n * dim].reshape(n, dim), [(r["u"], r["t"], r["d"]) for r in rows], [r["x"] for r in rows])
            self.seen.update(r["u"] for r in rows)
            self.next_doc = max(self.next_doc, max(r["d"] for r in rows) + 1)
        self.stats["live_pages"] = len({r["u"] for r in rows})

    # ------------------------------------------------------------------ live crawl parts (from embed_watch)
    def load_parts(self, parts_dir: str = PARTS_LIVE) -> int:
        """Append every finished embed_watch part not loaded yet. A part is finished once its offsets file exists
        and its .claim is gone (embed_watch writes the offsets last, then drops the claim)."""
        added = 0
        for off_path in sorted(glob.glob(os.path.join(parts_dir, "*.offsets.npy"))):
            prefix = off_path[:-len(".offsets.npy")]
            if prefix in self.parts_loaded or os.path.exists(prefix + ".claim"):
                continue
            t0 = time.perf_counter()
            vecs = np.load(prefix + ".npy")
            with open(prefix + ".meta.jsonl.zst", "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as r:
                meta = [orjson.loads(l) for l in io.TextIOWrapper(r, encoding="utf-8") if l.strip()]
            off = np.load(off_path)
            raw = open(prefix + ".passages.bin", "rb").read()
            keep, metas, texts, doc_of = [], [], [], {}
            with self.wlock:
                for j, m in enumerate(meta):
                    u = m["u"]
                    if u not in doc_of:  # first chunk of this page in the part: is the page new?
                        if u in self.seen:
                            doc_of[u] = None
                        else:
                            self.seen.add(u); doc_of[u] = self.next_doc; self.next_doc += 1
                    if doc_of[u] is None:
                        continue
                    keep.append(j); metas.append((u, m["t"], doc_of[u]))
                    texts.append(raw[off[j]:off[j + 1]].decode("utf-8", "ignore"))
                if keep:
                    self._append(vecs[keep], metas, texts)
            self.parts_loaded.add(prefix)
            self.stats["live_pages"] += sum(1 for v in doc_of.values() if v is not None)
            if keep:
                self.stats["last_ingest"] = time.time(); self.stats["ingest_unit"] = "embed_watch"
            added += len(keep)
        return added

    def tail_parts(self, interval: float = 1.0):
        last_t, last_n = time.time(), self.stats["live_passages"]
        while True:
            self.load_parts()
            now = time.time()
            if now - last_t >= 5:
                self.stats["ingest_pps"] = round((self.stats["live_passages"] - last_n) / (now - last_t), 1)
                last_t, last_n = now, self.stats["live_passages"]
            time.sleep(interval)


class Fetcher:
    """On-demand page fetch with aiohttp on a private event loop; same extraction as the crawler."""

    def __init__(self, timeout: float = 8.0, max_body: int = 20_000_000):
        # generous cap: big news homepages put megabytes of inline JSON before <body> (cnn.com: 3.4 MB), and a
        # truncated page parses to zero text
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="fetcher").start()
        self.timeout, self.max_body = timeout, max_body
        self.session = None
        self.robots = {}

    async def _sess(self):
        import aiohttp
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=aiohttp.ClientTimeout(total=self.timeout, connect=4))
        return self.session

    async def _get(self, url: str):
        s = await self._sess()
        try:
            async with s.get(url, allow_redirects=True, max_redirects=5) as r:
                if r.status != 200 or "html" not in r.headers.get("Content-Type", "html"):
                    return None
                body = bytearray()
                async for chunk in r.content.iter_chunked(1 << 16):
                    body += chunk
                    if len(body) > self.max_body:
                        break
                final = str(r.url)
        except Exception:
            return None
        title, _, text, links = extract(bytes(body), final, 300_000, 300)
        return {"url": final, "title": title, "text": text, "links": links}

    async def _allowed(self, url: str) -> bool:
        from urllib.parse import urlsplit
        from protego import Protego
        p = urlsplit(url); root = f"{p.scheme}://{p.netloc}"
        if root not in self.robots:
            s = await self._sess()
            try:
                async with s.get(root + "/robots.txt") as r:
                    self.robots[root] = Protego.parse(await r.text()) if r.status == 200 else None
            except Exception:
                self.robots[root] = None
        rb = self.robots[root]
        return rb is None or rb.can_fetch(url, USER_AGENT)

    def fetch(self, urls: list[str]) -> list[dict | None]:
        """Pages the user asked for, fetched directly like a browser would; None where a fetch failed."""
        async def go():
            return await asyncio.gather(*(self._get(u) for u in urls))
        return asyncio.run_coroutine_threadsafe(go(), self.loop).result()

    def expand(self, page: dict, n: int = 12) -> list[dict]:
        """One hop out from a fetched page: same-site links, robots.txt respected."""
        from urllib.parse import urlsplit
        host = urlsplit(page["url"]).netloc
        cand = [u for u in page["links"] if urlsplit(u).netloc == host and u != page["url"]][:n * 3]

        async def go():
            ok = [u for u in cand if await self._allowed(u)][:n]
            return await asyncio.gather(*(self._get(u) for u in ok))
        return [p for p in asyncio.run_coroutine_threadsafe(go(), self.loop).result() if p]


def clean_url(u: str) -> str | None:
    """Strip punctuation that ends the sentence around a pasted URL; a ')' stays when it closes a '(' in the URL
    (Wikipedia's Mercury_(planet))."""
    while u and u[-1] in ").,;'\">]":
        if u[-1] == ")" and u.count("(") >= u.count(")"):
            break
        u = u[:-1]
    return normalize_url(u)


def _seal_main(src: str, dst: str):
    """Build one HNSW segment (run as its own process: faiss + numpy only, all cores)."""
    import faiss
    x = np.load(src)
    idx = faiss.IndexHNSWFlat(x.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 100
    idx.add(x)
    faiss.write_index(idx, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "seal":
        _seal_main(sys.argv[2], sys.argv[3])
