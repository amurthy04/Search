"""On-device answer engine: crawl -> embed -> index -> retrieve -> Qwen3-0.6B, in one process.

  uv run python -m search.chat                      # http://localhost:8800
  uv run python -m search.chat --crawl              # also crawl the web live; new pages become searchable in seconds

Per turn: URLs in the message are fetched and indexed first; the message is embedded on the ANE (~0.7 ms) and
searched in HNSW (~0.15 ms); passages above the relevance threshold go into the prompt; the LLM (MLX, 4-bit, GPU)
prefills only the new turn on top of the session's KV cache and streams the answer.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

# torch is only needed to export CoreML graphs (search.embed export, run in a child process when one is missing).
# coremltools imports torch and scikit-learn whenever they are installed: seconds of startup, and torch brings its own
# OpenMP runtime next to faiss's.
for _m in ("torch", "sklearn"):
    sys.modules.setdefault(_m, None)

from .live import PARTS_LIVE, Fetcher, LiveIndex, clean_url  # noqa: E402
from .wikidata import wd_url  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # index and CoreML graph paths are repo-relative
# Weights are all local. Besides avoiding a network round trip at startup, this is required: a second
# AutoTokenizer.from_pretrained online revision check segfaults once a CoreML model is loaded in-process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# faiss and torch each bring an OpenMP runtime; a multithreaded faiss call segfaults inside libomp once both are
# loaded. faiss.omp_set_num_threads(1) only covers the calling thread, and HNSW adds run on ingest threads, so pin it
# process-wide before anything imports them.
os.environ["OMP_NUM_THREADS"] = "1"

SYSTEM = ("You are a helpful assistant that runs entirely on this computer, with a private web index built by its own "
          "crawler. When a <context> block is given, answer from it and cite sources inline as [1], [2]. If the context "
          "does not answer the question, say so in one sentence, then answer from general knowledge. Be concise.")
URL_RE = re.compile(r"https?://[^\s<>\"']+")
_POOL = ThreadPoolExecutor(4, thread_name_prefix="retrieve")  # side lookups that overlap the search (SQLite drops the GIL)
LIVE_CRAWL_DIR = "data/crawl/live"
LIVE_SHARDS = LIVE_CRAWL_DIR + "/*.jsonl.zst"


# ---------------------------------------------------------------------------- LLM worker
class LLM:
    """Owns every MLX call on one thread. Sessions keep their KV cache, so a turn prefills only its new tokens."""

    def __init__(self, path: str, gpu_busy: threading.Event, max_ctx: int = 24_000):
        self.path, self.max_ctx, self.gpu_busy = path, max_ctx, gpu_busy
        self.jobs: queue.Queue = queue.Queue()
        self.sessions: dict[str, dict] = {}
        self.ready = threading.Event()
        threading.Thread(target=self._run, daemon=True, name="llm").start()
        self.ready.wait()

    def _run(self):
        import mlx.core as mx
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        self.mx, self.stream_generate = mx, stream_generate
        self.make_cache, self.trim = make_prompt_cache, trim_prompt_cache
        t0 = time.perf_counter()
        self.model, self.tok = load(self.path)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

        def sampler(temp, top_p, k=20):
            # top-k over the 151k vocab first, then temperature + nucleus over just k candidates; mlx_lm's sampler
            # sorts the whole vocab for top-p every token (~7% slower decode)
            @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
            def f(logits):
                idx = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
                vals = mx.take_along_axis(logits, idx, axis=-1) / temp
                order = mx.argsort(-vals, axis=-1)
                vals = mx.take_along_axis(vals, order, axis=-1); idx = mx.take_along_axis(idx, order, axis=-1)
                probs = mx.softmax(vals, axis=-1)
                vals = mx.where(mx.cumsum(probs, axis=-1) - probs < top_p, vals, -mx.inf)
                return mx.take_along_axis(idx, mx.random.categorical(vals, axis=-1)[..., None], axis=-1).squeeze(-1)
            return f
        # Qwen3's recommended settings per mode
        self.samplers = {False: sampler(0.7, 0.8), True: sampler(0.6, 0.95)}
        self.greedy = lambda logits: mx.argmax(logits, axis=-1)  # deterministic decoding (evaluation)
        for think in (False, True):  # compile kernels + samplers for both modes before the first request
            c = self.make_cache(self.model)
            for _ in self.stream_generate(self.model, self.tok, self.tok.encode("Hello there, warm up."), max_tokens=4,
                                          sampler=self.samplers[think], prompt_cache=c):
                pass
        self.t_load = time.perf_counter() - t0
        self.ready.set()
        while True:
            self._do(self.jobs.get())

    def _turn_text(self, sess, user: str, think: bool) -> str:
        if sess["ids"]:
            head = "\n" if sess["ids"][-1] == self.im_end else "<|im_end|>\n"
        else:
            head = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        tail = "" if think else "<think>\n\n</think>\n\n"  # same as the chat template with enable_thinking=False
        return f"{head}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{tail}"

    def _do(self, job):
        out, sess = job["out"], self.sessions.setdefault(job["session"], {"ids": [], "cache": None})
        try:
            if sess["cache"] is None or len(sess["ids"]) > self.max_ctx:
                sess["ids"], sess["cache"] = [], self.make_cache(self.model)
            new = self.tok.encode(self._turn_text(sess, job["user"], job["think"]), add_special_tokens=False)
            sess["ids"] += new
            self.gpu_busy.set()
            t0 = time.perf_counter(); first = None; r = None
            sampler = self.greedy if job.get("greedy") else self.samplers[job["think"]]
            for r in self.stream_generate(self.model, self.tok, new, max_tokens=job["max_tokens"],
                                          sampler=sampler, prompt_cache=sess["cache"]):
                sess["ids"].append(r.token)
                if first is None:
                    first = time.perf_counter() - t0
                out.put({"t": r.text, "tps": round(r.generation_tps, 1)})
                if job["cancel"].is_set():
                    break
            # keep our token list identical to what the cache holds (an EOS can be fed without being yielded)
            off = sess["cache"][0].offset
            if off == len(sess["ids"]) + 1:
                sess["ids"].append(self.im_end)
            elif off > len(sess["ids"]):
                self.trim(sess["cache"], off - len(sess["ids"]))
            elif off < len(sess["ids"]):
                del sess["ids"][off:]
            out.put({"done": True, "n": r.generation_tokens if r else 0, "tps": round(r.generation_tps, 1) if r else 0,
                     "prefill_tokens": len(new), "prefill_tps": round(r.prompt_tps, 1) if r else 0,
                     "ttft_llm_ms": round((first or 0) * 1e3, 1), "ctx_tokens": len(sess["ids"]),
                     "finish": "stopped" if job["cancel"].is_set() else (r.finish_reason if r else None)})
        except Exception as e:  # never let one bad request kill the worker
            sess["cache"] = None
            out.put({"error": repr(e)})
        finally:
            self.gpu_busy.clear()
            out.put(None)
        if len(self.sessions) > 16:
            self.sessions.pop(next(iter(self.sessions)))

    def submit(self, session, user, think, max_tokens, cancel, greedy=False):
        out: queue.Queue = queue.Queue()
        self.jobs.put({"session": session, "user": user, "think": think, "max_tokens": max_tokens,
                       "cancel": cancel, "out": out, "greedy": greedy})
        return out

    def reset(self, session):
        self.sessions.pop(session, None)


# ---------------------------------------------------------------------------- live crawl
class Crawl:
    """Live crawl = two processes the app spawns: the crawler (CPU + network) and embed_watch (CoreML GPU), which
    turns each closed 300-page shard into an embedded part that LiveIndex.tail_parts picks up. Separate processes keep
    parsing/tokenizing off this process's GIL, which the LLM needs for every token."""

    def __init__(self):
        self.crawler, self.embedder = None, None

    def _spawn(self, args, log_name, env=None):
        log = os.open(f"{LIVE_CRAWL_DIR}/{log_name}", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        # posix_spawn, not fork: forking a process that holds Metal/CoreML/faiss state is unsafe. Children inherit
        # the raised fd limit and get their own process group so one signal reaches all their workers.
        pid = os.posix_spawn(sys.executable, [sys.executable, "-m", *args], env or os.environ,
                             file_actions=[(os.POSIX_SPAWN_DUP2, log, 1), (os.POSIX_SPAWN_DUP2, log, 2)], setpgroup=0)
        os.close(log)
        os.setpriority(os.PRIO_PGRP, pid, 10)  # background work: the LLM's token loop keeps the fast cores
        return pid

    def start(self, workers=4, n_seeds=20000, new_host_rate=40):
        os.makedirs(LIVE_CRAWL_DIR, exist_ok=True)
        if not self._alive("crawler"):
            # each run gets its own shard prefix (the crawler numbers shards from 0 every run) and the next block of
            # seed domains, so restarts reach new sites; seen-URL dedupe covers any overlap
            state_path = os.path.join(LIVE_CRAWL_DIR, "state.json")
            st = json.load(open(state_path)) if os.path.exists(state_path) else {"run": 0, "seed_offset": 100_000}
            run, seed_offset = st["run"] + 1, st["seed_offset"]
            json.dump({"run": run, "seed_offset": seed_offset + n_seeds}, open(state_path, "w"))
            self.crawler = self._spawn(["search.crawler", "--workers", str(workers), "--n-seeds", str(n_seeds),
                                        "--seed-offset", str(seed_offset), "--new-host-rate", str(new_host_rate),
                                        "--max-pages", "10000000", "--shard-pages", "300", "--out", f"{LIVE_CRAWL_DIR}/r{run}"],
                                       "crawler.log")
        self.ensure_embedder()

    def ensure_embedder(self):
        """(Re)start embed_watch if there are shards it hasn't embedded; it exits by itself once the crawl is over."""
        if not self._alive("embedder"):
            self.embedder = self._spawn(["search.embed_watch", "--backend", "coreml", "--batch-size", "16",
                                         "--shards", LIVE_SHARDS, "--parts", PARTS_LIVE],
                                        "embed_watch.log", env={**os.environ, "HF_HUB_OFFLINE": "1"})

    def stop(self):
        if self._alive("crawler"):
            os.killpg(self.crawler, signal.SIGCONT)
            os.killpg(self.crawler, signal.SIGINT)  # crawler flushes and closes its shards on SIGINT

    def shutdown(self):
        self.stop()
        if self._alive("embedder"):
            os.killpg(self.embedder, signal.SIGCONT); os.killpg(self.embedder, signal.SIGTERM)

    def pause_crawler(self, paused: bool):
        """Stop the crawler for the second it takes to fetch a page the user pasted: its ~2,000 open connections
        otherwise saturate the Wi-Fi link (the crawl's ceiling) and the user's fetch waits in line."""
        if self._alive("crawler"):
            try:
                os.killpg(self.crawler, signal.SIGSTOP if paused else signal.SIGCONT)
            except ProcessLookupError:
                pass

    def on_llm_busy(self, busy: bool):
        """Pause embedding while the LLM generates, so decode has the GPU (and memory bandwidth) to itself."""
        if self._alive("embedder"):
            try:
                os.killpg(self.embedder, signal.SIGSTOP if busy else signal.SIGCONT)
            except ProcessLookupError:
                pass
        for f in glob.glob("data/run/*.pid"):  # batch jobs started outside the app (search.wiki embed) register here
            try:
                pid = int(open(f).read())
                if self._is_batch_job(pid):
                    os.kill(pid, signal.SIGSTOP if busy else signal.SIGCONT)
            except (ValueError, OSError):
                pass

    _jobs: dict = {}

    def _is_batch_job(self, pid: int) -> bool:
        """pid is a live `python -m search.*` process (checked once per pid)."""
        if pid not in self._jobs:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
            self._jobs[pid] = "-m search." in cmd
        return self._jobs[pid]

    def _alive(self, which) -> bool:
        pid = getattr(self, which)
        if not pid:
            return False
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            done = pid
        if done:
            setattr(self, which, None)
        return not done

    def running(self):
        return self._alive("crawler")


# ---------------------------------------------------------------------------- answer pipeline
_REFER = set("it its it's they them their theirs this that these those he him his she her hers there then such".split())


def _followup(q: str) -> bool:
    """Does q lean on the previous turn? Very short ("and in 1990?", "why?") or refers back with a pronoun. A new
    standalone question in the same session is not retried with the old topic (that search is ~half of a turn's
    retrieval time, and its results are discarded)."""
    words = re.findall(r"[a-z']+", q.lower())
    return len(words) <= 4 or any(w in _REFER for w in words)


def retrieve(live, fetcher, crawl, message, prev_query, events, k_ctx=4, thr=0.70, ctx_chars=700, source_urls=False):
    """URLs -> fetch + embed (+ index if new); then search.
    Returns (context, sources, near misses, query, timings, URLs that yielded no readable text).

    thr: minimum cosine score for a passage to enter the context. 0.70 (was 0.75) puts the gold answer in the context
    for 65.9% of NQ dev questions (61.8% at 0.75) and lifts Qwen3-0.6B's answer accuracy 37.2% -> 38.5% (+102/-52
    questions of 3,610; bench/answer_eval.py). source_urls: each passage's URL in the prompt; the UI lists the sources itself, and
    without them the model stops pasting links into answers and prefills ~60 fewer tokens."""
    t, unread = {}, []
    urls = [u for u in (clean_url(x) for x in URL_RE.findall(message)) if u]
    got = []  # freshly fetched pages: [{url, title, texts, vecs, new}]
    if urls:
        events({"status": f"fetching {len(urls)} page(s)"})
        crawl.pause_crawler(True)
        try:
            t0 = time.perf_counter(); fetched = fetcher.fetch(urls); t["fetch_ms"] = (time.perf_counter() - t0) * 1e3
        finally:
            crawl.pause_crawler(False)
        pages = [p for p in fetched if p]
        t0 = time.perf_counter(); got = live.add_pages(pages, src="chat", max_chunks=60, rank_all=True)
        t["index_ms"] = (time.perf_counter() - t0) * 1e3
        read = {g["url"] for g in got}
        unread = [u for u, p in zip(urls, fetched) if not p or p["url"] not in read]
        for p in pages:  # one hop out, in the background: the answer doesn't wait for it
            threading.Thread(target=lambda p=p: live.add_pages(fetcher.expand(p), src="chat-expand"), daemon=True).start()
        events({"status": f"read {len(got)} page(s), {sum(len(g['texts']) for g in got)} passages"
                          if got else "could not read that page"})

    q = URL_RE.sub("", message).strip() or " ".join(g["title"] for g in got) or message
    if urls and not got:
        # the question was about pages we couldn't read; searching the index for "summarize this" only finds noise
        return (f"(The page(s) {', '.join(unread)} could not be read: the fetch failed or returned no text. Tell the user "
                "that, briefly, and do not describe or guess the page's content.)\n\n", [], [], q,
                {k: round(v, 2) for k, v in t.items()}, unread)
    def best(r):
        return max([h["score"] for h in r["hits"] + r.get("wiki", [])], default=0.0)
    # entity linking needs only the text: it runs while the query is embedded and searched
    link = _POOL.submit(live.cards.link, q) if not got else None
    res = live.search(q, k=8, snippets=False)
    if prev_query and not got and best(res) < thr and _followup(q):
        # likely a follow-up ("how is it treated?"): retry with the previous topic, keep whichever matches better
        res2 = live.search(f"{prev_query} {q}", k=8, snippets=False)
        if best(res2) > best(res):
            res, q = res2, f"{prev_query} {q}"
            link = _POOL.submit(live.cards.link, q)
    t["embed_ms"], t["hnsw_ms"], t["search_ms"] = res["t_ms"]["embed"], res["t_ms"]["search"], res["t_ms"]["total"]
    t["wiki_ms"] = res["t_ms"].get("wiki", 0.0)

    picked = []  # (score, url, title, text, live)
    if got:  # the user pointed at these pages: always include their best chunks
        qv = res["qv"]
        for g in got:
            sc = g["vecs"] @ qv
            picked += [(float(sc[i]), g["url"], g["title"], g["texts"][i], "live") for i in np.argsort(-sc)[:3]]
    cap = k_ctx + (2 if got else 0)  # context blocks, pasted pages' chunks first
    picked = picked[:cap]
    seen = {p[1] for p in picked}
    # web and Wikipedia hits come from the same encoder, so their cosine scores rank against each other directly
    cands = [(h["score"], h["url"], h["title"], h["pid"], "live" if h["pid"] >= live.base_n else "web")
             for h in res["hits"]]
    cands += [(h["score"], h["url"], h["title"], h["pid"], "wikidata" if "wikidata.org" in h["url"] else "wiki")
              for h in res.get("wiki", [])]
    chosen = []
    for c in sorted(cands, key=lambda c: -c[0]):
        if c[0] >= thr and c[1] not in seen:
            seen.add(c[1]); chosen.append(c)
    chosen = chosen[:cap - len(picked)]
    # request every chosen passage's pages at once, then read them
    live.prefetch_passages([c[3] for c in chosen if c[4] in ("web", "live")], ctx_chars)
    live.wiki.prefetch_passages([c[3] for c in chosen if c[4] in ("wiki", "wikidata")])
    for sc, url, title, pid, kind in chosen:
        text = live.wiki.passage(pid, ctx_chars) if kind in ("wiki", "wikidata") else live.passage(pid, ctx_chars)
        picked.append((sc, url, title, text, kind))
    # Wikidata: entities named in the question get their fact card even if no passage about them matched
    linked = [] if got else [tl for tl in link.result() if tl not in {p[2] for p in picked}]

    sources, blocks = [], []
    carded = {p[2] for p in picked if p[4] == "wikidata"}
    # a Wikipedia hit carries its article's facts, unless the card itself is already in the context: look the
    # cards up in parallel (each pool thread has its own SQLite connection)
    row_of = {h["title"]: h.get("doc") for h in res.get("wiki", [])}
    want = []
    for _, _, title, _, kind in picked:
        if kind == "wiki" and title not in carded and title not in want:
            want.append(title)
    cards = dict(zip(want, live.cards.cards_for([row_of.get(t) for t in want], want)))
    linked_cards = dict(zip(linked[:2], _POOL.map(live.cards.get, linked[:2])))
    for n, (s, url, title, text, kind) in enumerate(picked, 1):
        art = title
        if kind in ("wiki", "wikidata") and "\n" in text:
            head, text = text.split("\n", 1)  # "Title — Section" / "Title — Wikidata facts" header line
            title = head
        text = text[len(title):].lstrip() if title and text.startswith(title) else text
        text = text[:ctx_chars]
        card = cards.get(art) if kind == "wiki" and art not in carded else None
        carded.add(art)
        sources.append({"n": n, "url": url, "title": title or url, "score": round(s, 3), "snippet": text[:240],
                        "live": kind == "live", "kind": kind})
        blocks.append(f"[{n}] {title}" + (f" ({url})" if source_urls else "") + f"\n{text}" + (f"\nWikidata: {card[:400]}" if card else ""))
    for tl in linked[:2]:
        if tl in carded:
            continue
        got_card = linked_cards.get(tl)
        if got_card:
            qid, card = got_card
            n = len(sources) + 1
            sources.append({"n": n, "url": wd_url(qid), "title": tl, "score": 1.0, "snippet": card[:240], "live": False,
                            "kind": "wikidata"})
            blocks.append(f"[{n}] {tl} (Wikidata facts)\n{card[:600]}")
    near = [{"url": h["url"], "title": h["title"], "score": round(h["score"], 3)}
            for h in sorted(res["hits"] + res.get("wiki", []), key=lambda h: -h["score"])[:3]]
    ctx = "<context>\n" + "\n\n".join(blocks) + "\n</context>\n\n" if blocks else ""
    # tell the model what it does not have, or a 0.6B model will happily cite a [1] that isn't there
    if unread:
        ctx += f"(Note: these pages could not be read: {', '.join(unread)}. Say so.)\n"
    if not blocks:
        ctx += "(No search results matched this question. Answer from general knowledge and do not cite sources.)\n\n"
    return ctx, sources, near, q, {k: round(v, 2) for k, v in t.items()}, unread


def make_handler(llm, live, fetcher, crawl, args):
    page = os.path.join(ROOT, "search", "web", "chat.html")
    prev_q: dict[str, str] = {}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n)) if n else {}

        def do_GET(self):
            if self.path == "/":
                b = open(page, "rb").read()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
            elif self.path == "/status":
                self._json({"base_passages": live.base_n, "passages": live.size, **live.status(),
                            "crawling": crawl.running(), "model": os.path.basename(args.model)})
            elif self.path.startswith("/search?"):
                from urllib.parse import parse_qs, urlsplit
                q = parse_qs(urlsplit(self.path).query).get("q", [""])[0]
                r = live.search(q, 10); r.pop("qv", None)
                self._json(r)
            else:
                self.send_error(404)

        def do_POST(self):
            # a page on another site can POST "simple" requests here without a preflight: accept JSON from our own
            # origin only (a cross-site JSON request needs a CORS preflight, which this server never grants)
            origin = self.headers.get("Origin")
            if (not (self.headers.get("Content-Type") or "").startswith("application/json")
                    or (origin and origin not in (f"http://localhost:{args.port}", f"http://127.0.0.1:{args.port}"))):
                return self.send_error(403)
            if self.path == "/crawl":
                body = self._body()
                crawl.start(workers=min(16, max(1, int(body.get("workers", 4))))) if body.get("on") else crawl.stop()
                return self._json({"crawling": crawl.running()})
            if self.path == "/reset":
                llm.reset(self._body().get("session", "")); return self._json({"ok": True})
            if self.path != "/chat":
                return self.send_error(404)
            req = self._body()
            sid, msg = req.get("session") or uuid.uuid4().hex, req["message"]
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close"); self.end_headers()
            t_req = time.perf_counter()

            def send(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()

            cancel, out = threading.Event(), None
            crawl.on_llm_busy(True)  # the whole turn (URL embedding + generation) gets the GPU; ingest waits
            try:
                if req.get("search", True):
                    prev = prev_q.get(sid)
                    ctx, sources, near, q, t, unread = retrieve(live, fetcher, crawl, msg, prev, send)
                    # a follow-up searched "<topic> <question>": the topic stays, so the query doesn't grow every turn
                    prev_q[sid] = prev if prev and q.startswith(prev + " ") else q
                    send({"sources": sources, "near": near, "query": q, "t": t, "unread": unread})
                    user = f"{ctx}{msg}"
                else:
                    user = msg
                t_llm = time.perf_counter()
                out = llm.submit(sid, user, bool(req.get("think")), int(req.get("max_tokens", 2048)), cancel)
                first = True
                while (ev := out.get()) is not None:
                    if first and "t" in ev:
                        ev["ttft_ms"] = round((time.perf_counter() - t_req) * 1e3, 1); first = False
                    if ev.get("done"):
                        ev["retrieval_ms"] = round((t_llm - t_req) * 1e3, 1)
                    send(ev)
            except (BrokenPipeError, ConnectionResetError):
                cancel.set()  # Stop button: the LLM thread ends this turn at the next token
                while out is not None and (ev := out.get()) is not None:
                    pass
            finally:
                crawl.on_llm_busy(False)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-mlx-4bit"))
    ap.add_argument("--emb", default="data/index/emb_all")
    ap.add_argument("--index", default="data/index/idx_all.hnsw")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--crawl", action="store_true", help="start a live crawl on launch")
    args = ap.parse_args()

    t0 = time.perf_counter()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  # the crawler holds thousands of sockets
    want = min(65536, int(subprocess.run(["sysctl", "-n", "kern.maxfilesperproc"], capture_output=True, text=True).stdout or 10240))
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, want), hard))
    # transformers is imported by both the LLM thread (mlx_lm) and the index loader (the tokenizer); its lazy module
    # raises ImportError when two threads initialize it at once, so finish that import here first
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: F401
    llm_box, gpu_busy = {}, threading.Event()
    th = threading.Thread(target=lambda: llm_box.setdefault("llm", LLM(args.model, gpu_busy)))  # load while the index loads
    th.start()
    live = LiveIndex(args.emb, args.index, gpu_busy=gpu_busy)
    th.join(); llm = llm_box["llm"]
    fetcher = Fetcher()
    crawl = Crawl()
    if glob.glob(LIVE_SHARDS):
        crawl.ensure_embedder()  # embed any shards left over from the last crawl
    threading.Thread(target=live.tail_parts, daemon=True, name="parts").start()
    threading.Thread(target=live.keep_warm, daemon=True, name="warm").start()
    threading.Thread(target=live.wiki_reload, daemon=True, name="wiki").start()
    if args.crawl:
        crawl.start()
    print(f"ready in {time.perf_counter() - t0:.1f}s: {live.size:,} passages (index {live.t_load:.1f}s, "
          f"LLM {llm.t_load:.1f}s) -> http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer.request_queue_size = 128  # default 5: a burst of requests gets connection resets
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(llm, live, fetcher, crawl, args))

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)  # so `kill` also stops the crawler and embedder
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        crawl.on_llm_busy(False)  # a turn cut short by shutdown never ran its own SIGCONT
        crawl.shutdown()


if __name__ == "__main__":
    main()
