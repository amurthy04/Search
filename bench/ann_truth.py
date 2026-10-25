"""Exact nearest neighbours for a query set, by brute force on the GPU (MLX), as ground truth for ANN tuning.

  uv run python bench/ann_truth.py encode --qs data/eval/nq_open_dev.jsonl --out data/eval/nq
  uv run python bench/ann_truth.py exact --q data/eval/nq.qv.npy --out data/eval/nq

encode: query vectors from the chat's own query graph (CoreML, ANE, batch 1, 32 tokens) -> <out>.qv.npy
exact:  top-100 over the Wikipedia vectors (tombstoned rows skipped) and the web vectors -> <out>.{wiki,web}.{ids,scores}.npy
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402

K = 100


def cmd_encode(a):
    from search.embed import make_embedder
    qs = [json.loads(l)["q"] for l in open(a.qs)]
    emb = make_embedder("coreml_ane", "BAAI/bge-small-en-v1.5", max_seq=a.seq, batch_size=1)
    for q in qs[:3]:
        emb.encode([q], is_query=True)
    t0 = time.perf_counter()
    qv = np.concatenate([emb.encode([q], is_query=True) for q in qs]).astype(np.float32)
    print(f"{len(qs)} queries in {time.perf_counter() - t0:.1f}s")
    np.save(a.out + ".qv.npy", qv)


def topk_exact(q: np.ndarray, rows, n: int, chunk: int, dead: np.ndarray | None = None):
    """rows(a, b) -> fp16 array of rows [a, b). Returns (ids, scores), each (nq, K), best first."""
    import mlx.core as mx
    Q = mx.array(q.astype(np.float16))
    best_s = np.full((len(q), K), -np.inf, np.float32); best_i = np.full((len(q), K), -1, np.int64)
    t0 = time.perf_counter()
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        X = mx.array(np.ascontiguousarray(rows(a, b)))
        S = (Q @ X.T).astype(mx.float32)
        if dead is not None and dead[a:b].any():
            S = S + mx.array(np.where(dead[a:b] > 0, -1e9, 0.0).astype(np.float32))[None, :]
        I = mx.argpartition(-S, K - 1, axis=1)[:, :K]
        V = mx.take_along_axis(S, I, axis=1)
        mx.eval(I, V)
        I = np.array(I).astype(np.int64) + a; V = np.array(V)
        cs = np.concatenate([best_s, V], 1); ci = np.concatenate([best_i, I], 1)
        o = np.argpartition(-cs, K - 1, axis=1)[:, :K]
        best_s = np.take_along_axis(cs, o, 1); best_i = np.take_along_axis(ci, o, 1)
        if (a // chunk) % 10 == 0:
            print(f"  {b:,}/{n:,} rows  {time.perf_counter() - t0:.0f}s", flush=True)
    o = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_i, o, 1), np.take_along_axis(best_s, o, 1)


def cmd_exact(a):
    q = np.load(a.q)
    st = json.load(open("data/wiki/index/wiki.state.json"))
    n = st["n"]
    vecs = np.memmap("data/wiki/index/wiki.f16", dtype=np.float16, mode="r", shape=(n, 384))
    dead = np.fromfile("data/wiki/index/wiki.dead.u8", np.uint8)
    dead = np.concatenate([dead, np.zeros(max(0, n - len(dead)), np.uint8)])[:n]
    print(f"wiki: {n:,} rows ({int(dead.sum()):,} dead)")
    ids, sc = topk_exact(q, lambda x, y: vecs[x:y], n, a.chunk, dead)
    np.save(a.out + ".wiki.ids.npy", ids); np.save(a.out + ".wiki.scores.npy", sc)
    np.save(a.out + ".wiki.n.npy", np.array([n]))
    web = np.load("data/index/emb_all.npy", mmap_mode="r")
    print(f"web: {len(web):,} rows")
    ids, sc = topk_exact(q, lambda x, y: web[x:y], len(web), a.chunk)
    np.save(a.out + ".web.ids.npy", ids); np.save(a.out + ".web.scores.npy", sc)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--qs", required=True); e.add_argument("--out", required=True)
    e.add_argument("--seq", type=int, default=32)
    x = sub.add_parser("exact"); x.add_argument("--q", required=True); x.add_argument("--out", required=True)
    x.add_argument("--chunk", type=int, default=262_144)
    a = ap.parse_args()
    {"encode": cmd_encode, "exact": cmd_exact}[a.cmd](a)


if __name__ == "__main__":
    main()
