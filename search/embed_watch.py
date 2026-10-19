"""Embed crawl shards as they close, while the crawl is still running.

Each shard becomes one 'part' (vectors + meta + passages). Several watchers can run at once on different compute
units (e.g. CoreML GPU and ANE) by taking disjoint shards (--mod N --rem r). `search.merge_parts` joins the parts.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np
import orjson
import zstandard as zstd

from .corpus import iter_passages


def crawler_running() -> bool:
    return subprocess.run(["pgrep", "-f", "search.crawler"], capture_output=True).returncode == 0


def closed_shards(pattern: str) -> list[str]:
    """All shards except the newest one per worker while the crawler is alive (that one is still being written)."""
    shards = sorted(glob.glob(pattern))
    if not crawler_running():
        return shards
    newest = {}
    for s in shards:
        w = os.path.basename(s).split(".")[1]  # pages.w00.0003.jsonl.zst -> w00
        if w not in newest or os.path.getmtime(s) > os.path.getmtime(newest[w]):
            newest[w] = s
    return [s for s in shards if s not in newest.values()]


def embed_shard(emb, shard: str, out_prefix: str, args) -> dict:
    t0 = time.perf_counter()
    texts, meta = [], []
    for doc_id, ci, url, title, passage in iter_passages(shard, args.words, args.overlap, args.max_chunks):
        texts.append(passage); meta.append({"d": doc_id, "c": ci, "u": url, "t": title})
    if not texts:
        return {"passages": 0}
    order = np.argsort([len(t) for t in texts])
    vecs = np.empty((len(texts), 0), np.float16)
    out = []
    for i in range(0, len(order), args.flush):
        idx = order[i:i + args.flush]
        out.append((idx, emb.encode([texts[j] for j in idx]).astype(np.float16)))
    dim = out[0][1].shape[1]
    vecs = np.empty((len(texts), dim), np.float16)
    for idx, v in out:
        vecs[idx] = v
    np.save(out_prefix + ".npy", vecs)
    with open(out_prefix + ".meta.jsonl.zst", "wb") as fh:
        w = zstd.ZstdCompressor(level=3).stream_writer(fh)
        for m in meta:
            w.write(orjson.dumps(m) + b"\n")
        w.close()
    offsets = [0]
    with open(out_prefix + ".passages.bin", "wb") as fh:
        for t in texts:
            b = t.encode("utf-8"); fh.write(b); offsets.append(offsets[-1] + len(b))
    np.save(out_prefix + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    dt = time.perf_counter() - t0
    return {"passages": len(texts), "s": dt, "per_s": len(texts) / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/crawl_main/*.jsonl.zst")
    ap.add_argument("--parts", default="data/index/parts")
    ap.add_argument("--backend", default="coreml"); ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--flush", type=int, default=4096)
    ap.add_argument("--words", type=int, default=180); ap.add_argument("--overlap", type=int, default=30); ap.add_argument("--max-chunks", type=int, default=6)
    ap.add_argument("--mod", type=int, default=1); ap.add_argument("--rem", type=int, default=0)
    ap.add_argument("--once", action="store_true", help="process what is closed now and exit (no waiting for the crawler)")
    args = ap.parse_args()
    from .embed import make_embedder
    emb = make_embedder(args.backend, args.model, max_seq=args.max_seq, batch_size=args.batch_size)
    os.makedirs(args.parts, exist_ok=True)
    tag = args.backend
    total = 0; t_start = time.time()
    while True:
        todo = []
        for i, s in enumerate(closed_shards(args.shards)):
            if i % args.mod != args.rem:
                continue
            name = os.path.basename(s).replace(".jsonl.zst", "")
            out_prefix = os.path.join(args.parts, name)
            if os.path.exists(out_prefix + ".npy") or os.path.exists(out_prefix + ".claim"):
                continue
            todo.append((s, out_prefix))
        if not todo:
            if args.once or not crawler_running():
                break
            time.sleep(15)
            continue
        for s, out_prefix in todo:
            open(out_prefix + ".claim", "w").write(tag)
            r = embed_shard(emb, s, out_prefix, args)
            os.remove(out_prefix + ".claim")
            total += r.get("passages", 0)
            print(orjson.dumps({"shard": os.path.basename(s), "backend": tag, **r, "total": total,
                                "elapsed_min": (time.time() - t_start) / 60}).decode(), flush=True)
    print(f"done: {total} passages in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
