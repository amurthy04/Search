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


def load_meta(prefix: str):
    urls, titles, docs = [], [], []
    with open(prefix + ".meta.jsonl.zst", "rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8"):
                if line.strip():
                    m = orjson.loads(line)
                    urls.append(m["u"]); titles.append(m["t"]); docs.append(m["d"])
    return urls, titles, np.asarray(docs, dtype=np.int64)


class Passages:
    """Random access to passage text via mmap + offsets (written by search.embed run)."""

    def __init__(self, prefix: str):
        self.ok = os.path.exists(prefix + ".passages.bin")
        if self.ok:
            self.off = np.load(prefix + ".offsets.npy")
            self.mm = np.memmap(prefix + ".passages.bin", dtype=np.uint8, mode="r")

    def get(self, i: int, max_chars: int = 300) -> str:
        if not self.ok:
            return ""
        a, b = int(self.off[i]), int(self.off[i + 1])
        return self.mm[a:min(b, a + max_chars * 4)].tobytes().decode("utf-8", "ignore")[:max_chars]


class Engine:
    def __init__(self, model: str, emb_prefix: str, index_path: str, kind: str, backend: str = "st", device: str = "mps",
                 ef: int = 64, threads: int = 1, max_seq: int = 64):
        from .embed import make_embedder
        from .index import Searcher
        t0 = time.perf_counter()
        self.emb = make_embedder(backend, model, device=device, fp16=(device != "cpu"), max_seq=max_seq, batch_size=32)
        self.t_model = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.searcher = Searcher(kind, index_path, threads=threads)
        self.searcher.set_param(ef)
        self.t_index = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.urls, self.titles, self.docs = load_meta(emb_prefix)
        self.passages = Passages(emb_prefix)
        self.t_meta = time.perf_counter() - t0
        self.emb.encode(["warmup query"], is_query=True)

    def search(self, query: str, k: int = 10, overfetch: int = 3, snippets: bool = True) -> dict:
        t0 = time.perf_counter()
        qv = self.emb.encode([query], is_query=True).astype(np.float32)
        t1 = time.perf_counter()
        scores, ids = self.searcher.search(qv, k * overfetch)
        t2 = time.perf_counter()
        seen = set(); hits = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            d = int(self.docs[i])
            if d in seen:
                continue
            seen.add(d)
            hits.append({"score": float(s), "url": self.urls[i], "title": self.titles[i], "pid": int(i)})
            if len(hits) >= k:
                break
        t3 = time.perf_counter()
        if snippets:
            for h in hits:
                h["snippet"] = self.passages.get(h["pid"])
        t4 = time.perf_counter()
        return {"query": query, "hits": hits, "t_ms": {"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3,
                "dedupe": (t3 - t2) * 1e3, "snippets": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}}


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
        app = web.Application(); app.router.add_get("/search", handle)
        web.run_app(app, port=args.port, print=lambda *a: print(f"serving on :{args.port}", flush=True))


if __name__ == "__main__":
    main()
