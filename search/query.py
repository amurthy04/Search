"""Query path: embed query -> ANN search -> dedupe by document -> snippets. Every stage timed.

Usage:
  python -m search.query repl --model ... --index data/index/idx.hnsw
  python -m search.query serve --port 8080
  python -m search.query ask "how do transformers work"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import orjson
import zstandard as zstd


import re as _re
_title_sep = _re.compile(r"\s*[|\-–—:]\s*")


def _norm_title(t: str) -> str:
    """Title without the site-name suffix/prefix, lowercased: 'Foo Bar – SiteName' -> 'foo bar'."""
    parts = [p for p in _title_sep.split(t.lower().strip()) if len(p) > 12]
    return max(parts, key=len) if parts else t.lower().strip()


class StrTable:
    """Read-only list of strings as one byte blob + offsets: 2.4M URLs as Python strings hold ~300 MB and take seconds
    to build; as one bytes object they hold their UTF-8 size and load in ~0.1 s. Items appended later (live passages)
    go to a plain list."""

    def __init__(self, path: str):
        self.off = np.load(path + ".off.npy")
        # read into RAM, not mmapped: every query reads a few dozen random entries, and under memory pressure each
        # cold page of an mmap is a separate SSD read
        self.mm = open(path + ".bin", "rb").read()
        self.n = len(self.off) - 1
        self.extra: list[str] = []
        self.off_path = path + ".off.npy"

    @staticmethod
    def write(path: str, items):
        off = [0]
        with open(path + ".bin.tmp", "wb") as f:
            for s in items:
                b = s.encode("utf-8"); f.write(b); off.append(off[-1] + len(b))
        np.save(path + ".off.tmp.npy", np.asarray(off, np.int64))
        os.replace(path + ".bin.tmp", path + ".bin"); os.replace(path + ".off.tmp.npy", path + ".off.npy")

    def __len__(self):
        return self.n + len(self.extra)

    def __getitem__(self, i: int) -> str:
        if i >= self.n:
            return self.extra[i - self.n]
        return self.mm[int(self.off[i]):int(self.off[i + 1])].decode("utf-8")

    def append(self, s: str):
        self.extra.append(s)

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_meta(prefix: str):
    """(urls, titles, doc ids) of every passage. Parsed once from <prefix>.meta.jsonl.zst, then cached as string
    tables next to it (2.4M JSON lines take ~7 s to parse; the tables load in ~10 ms)."""
    src = prefix + ".meta.jsonl.zst"
    cache = prefix + ".meta"
    if not (os.path.exists(cache + ".docs.npy") and os.path.getmtime(cache + ".docs.npy") >= os.path.getmtime(src)):
        urls, titles, docs = [], [], []
        with open(src, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as r:
                for line in io.TextIOWrapper(r, encoding="utf-8"):
                    if line.strip():
                        m = orjson.loads(line)
                        urls.append(m["u"]); titles.append(m["t"]); docs.append(m["d"])
        StrTable.write(cache + ".urls", urls); StrTable.write(cache + ".titles", titles)
        np.save(cache + ".docs.tmp.npy", np.asarray(docs, dtype=np.int64)); os.replace(cache + ".docs.tmp.npy", cache + ".docs.npy")
    return StrTable(cache + ".urls"), StrTable(cache + ".titles"), np.load(cache + ".docs.npy")


def in_sorted(base: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Which of the hashes h are in the sorted hash array base."""
    if not len(base):
        return np.zeros(len(h), bool)
    return base[np.minimum(np.searchsorted(base, h), len(base) - 1)] == h


class UrlSet:
    """Membership of URLs: the base set as sorted 64-bit hashes (a numpy array, 8 bytes per URL), additions in a
    Python set. Replaces set(urls) over millions of base URLs (~300 MB, seconds to build)."""

    def __init__(self, urls: "StrTable", cache: str | None = None):
        import xxhash
        self._h = xxhash.xxh3_64_intdigest
        if cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(urls.off_path):
            self.base = np.load(cache)
        else:
            self.base = np.sort(np.fromiter((self._h(u.encode()) for u in urls), np.uint64, len(urls)))
            if cache:
                np.save(cache, self.base)
        self.extra: set[str] = set()

    def __contains__(self, u: str) -> bool:
        return u in self.extra or bool(in_sorted(self.base, np.array([self._h(u.encode())], np.uint64))[0])

    def add(self, u: str):
        self.extra.add(u)

    def update(self, us):
        self.extra.update(us)


class Passages:
    """Random access to passage text via mmap + offsets (written by search.embed run)."""

    def __init__(self, prefix: str):
        self.ok = os.path.exists(prefix + ".passages.bin")
        if self.ok:
            from .ann import Blob
            self.off = np.load(prefix + ".offsets.npy")
            self.blob = Blob(prefix + ".passages.bin")

    def _range(self, i: int, max_chars: int) -> tuple[int, int]:
        a = int(self.off[i])
        return a, min(int(self.off[i + 1]), a + max_chars * 4) - a

    def prefetch(self, ids, max_chars: int = 700):
        """Ask for these passages' pages all at once (random reads issued together, not one after another)."""
        if self.ok:
            self.blob.prefetch(self._range(i, max_chars) for i in ids)

    def get(self, i: int, max_chars: int = 300) -> str:
        if not self.ok:
            return ""
        return self.blob.read(*self._range(i, max_chars)).decode("utf-8", "ignore")[:max_chars]


class Engine:
    def __init__(self, model: str, emb_prefix: str, index_path: str, kind: str, backend: str = "st", device: str = "mps",
                 ef: int = 64, threads: int = 1, max_seq: int = 64, extra_seqs: tuple[int, ...] = ()):
        """kind="none": no ANN index is loaded (the caller searches the vectors some other way and uses collect())."""
        from .embed import make_embedder
        from .index import Searcher
        t0 = time.perf_counter()
        bs = 1 if backend.startswith("coreml") else 32  # CoreML graph is fixed-shape: batch 1 for single queries
        self.emb = make_embedder(backend, model, device=device, fp16=(device != "cpu"), max_seq=max_seq, batch_size=bs,
                                 **({"extra_seqs": extra_seqs} if extra_seqs else {}))
        self.t_model = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.searcher = None
        if kind != "none":
            self.searcher = Searcher(kind, index_path, threads=threads)
            self.searcher.set_param(ef)
        self.t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.urls, self.titles, self.docs = load_meta(emb_prefix)
        self.passages = Passages(emb_prefix)
        self.t_meta = time.perf_counter() - t0
        self.emb.encode(["warmup query"], is_query=True)

    def embed(self, query: str) -> np.ndarray:
        return self.emb.encode([query], is_query=True).astype(np.float32)[0]

    def search(self, query: str, k: int = 10, overfetch: int = 3, snippets: bool = True,
               qv: np.ndarray | None = None) -> dict:
        """qv: the query's vector if the caller already has it (it searches a second index with the same one)."""
        t0 = time.perf_counter()
        qv = (self.embed(query) if qv is None else qv)[None, :]
        t1 = time.perf_counter()
        scores, ids = self.searcher.search(qv, k * overfetch)
        t2 = time.perf_counter()
        res = self.collect(query, scores[0], ids[0], k, snippets)
        res["t_ms"] = {"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3, **res["t_ms"]}
        res["t_ms"]["total"] = (time.perf_counter() - t0) * 1e3
        return res

    def collect(self, query: str, scores, ids, k: int = 10, snippets: bool = True) -> dict:
        """Best-first (score, passage id) candidates -> hits: one per document, syndicated copies collapsed."""
        t2 = time.perf_counter()
        seen = set(); seen_titles = set(); hits = []
        for s, i in zip(scores, ids):
            if i < 0:
                continue
            d = int(self.docs[i])
            if d in seen:
                continue
            tkey = _norm_title(self.titles[i])
            if tkey and tkey in seen_titles:
                continue  # syndicated copy of a page already shown (same title) -> collapse
            seen.add(d); seen_titles.add(tkey)
            hits.append({"score": float(s), "url": self.urls[i], "title": self.titles[i], "pid": int(i)})
            if len(hits) >= k:
                break
        t3 = time.perf_counter()
        if snippets:
            for h in hits:
                h["snippet"] = self.passages.get(h["pid"])
        t4 = time.perf_counter()
        return {"query": query, "hits": hits, "t_ms": {"dedupe": (t3 - t2) * 1e3, "snippets": (t4 - t3) * 1e3}}


def fmt(res: dict) -> str:
    out = [f"q: {res['query']}   " + " ".join(f"{k}={v:.1f}ms" for k, v in res["t_ms"].items())]
    for i, h in enumerate(res["hits"], 1):
        out.append(f"{i:2d}. [{h['score']:.3f}] {h['title'][:80]}\n     {h['url']}")
        if h.get("snippet"):
            out.append("     " + h["snippet"][:200].replace("\n", " "))
    return "\n".join(out)


def add_common(p):
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--backend", default="st")
    p.add_argument("--device", default="mps")
    p.add_argument("--emb", default="data/index/emb")
    p.add_argument("--index", default="data/index/idx.hnsw")
    p.add_argument("--kind", default="hnsw")
    p.add_argument("--ef", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--k", type=int, default=10)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("repl", "ask", "serve"):
        p = sub.add_parser(name); add_common(p)
        if name == "ask":
            p.add_argument("query", nargs="+")
        if name == "serve":
            p.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    eng = Engine(args.model, args.emb, args.index, args.kind, args.backend, args.device, args.ef, args.threads)
    print(f"loaded: model {eng.t_model:.1f}s index {eng.t_index:.1f}s meta {eng.t_meta:.1f}s; {len(eng.urls)} passages", flush=True)
    if args.cmd == "ask":
        print(fmt(eng.search(" ".join(args.query), args.k)))
    elif args.cmd == "repl":
        while True:
            try:
                q = input("\nsearch> ").strip()
            except EOFError:
                break
            if q:
                print(fmt(eng.search(q, args.k)))
    else:
        from aiohttp import web

        async def handle(req):
            q = req.query.get("q", "")
            k = int(req.query.get("k", args.k))
            return web.json_response(eng.search(q, k))
        page = os.path.join(os.path.dirname(__file__), "web", "index.html")

        async def home(req):
            return web.FileResponse(page)
        app = web.Application(); app.router.add_get("/search", handle); app.router.add_get("/", home)
        web.run_app(app, port=args.port, print=lambda *a: print(f"serving on :{args.port}", flush=True))


if __name__ == "__main__":
    main()
