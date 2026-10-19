"""Build and benchmark vector indexes over the embedded corpus.

Index types:
  flat     exact inner-product search (faiss IndexFlatIP) - the recall ground truth
  hnsw     faiss IndexHNSWFlat (M, efConstruction, efSearch)
  hnsw_sq  faiss HNSW over 8-bit scalar-quantized vectors (4x smaller, ~same recall)
  usearch  usearch HNSW (f16 / i8 quantization), multithreaded build
  ivf      faiss IVF-Flat (nlist, nprobe)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def load_vectors(prefix: str) -> np.ndarray:
    x = np.load(prefix + ".npy")
    return np.ascontiguousarray(x.astype(np.float32))


def build(kind: str, x: np.ndarray, out: str, M: int = 32, efc: int = 200, nlist: int = 0, threads: int = 0):
    import faiss
    if threads:
        faiss.omp_set_num_threads(threads)
    n, d = x.shape
    t0 = time.perf_counter()
    if kind == "flat":
        idx = faiss.IndexFlatIP(d)
        idx.add(x)
        faiss.write_index(idx, out)
    elif kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efc
        idx.add(x)
        faiss.write_index(idx, out)
    elif kind == "hnsw_sq":
        idx = faiss.IndexHNSWSQ(d, faiss.ScalarQuantizer.QT_8bit, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efc
        idx.train(x)
        idx.add(x)
        faiss.write_index(idx, out)
    elif kind == "ivf":
        nlist = nlist or int(4 * np.sqrt(n))
        q = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(x[np.random.default_rng(0).choice(n, min(n, nlist * 64), replace=False)])
        idx.add(x)
        faiss.write_index(idx, out)
    elif kind.startswith("usearch"):
        from usearch.index import Index, ScalarKind
        dtype = {"usearch": ScalarKind.F32, "usearch_f16": ScalarKind.F16, "usearch_i8": ScalarKind.I8}[kind]
        idx = Index(ndim=d, metric="ip", dtype=dtype, connectivity=M, expansion_add=efc)
        idx.add(np.arange(n, dtype=np.uint64), x, threads=threads or 0)
        idx.save(out)
    else:
        raise ValueError(kind)
    dt = time.perf_counter() - t0
    size = os.path.getsize(out)
    return {"kind": kind, "n": int(n), "dim": int(d), "build_s": dt, "size_mb": size / 1e6, "M": M, "efc": efc, "nlist": nlist}


class Searcher:
    """Uniform search wrapper with a tunable search-time parameter."""

    def __init__(self, kind: str, path: str, threads: int = 1):
        self.kind = kind
        if kind.startswith("usearch"):
            from usearch.index import Index
            self.idx = Index.restore(path)
            self.threads = threads
        else:
            import faiss
            faiss.omp_set_num_threads(threads)
            self.idx = faiss.read_index(path)
            self.faiss = faiss

    def set_param(self, v: int):
        if self.kind.startswith("hnsw"):
            self.idx.hnsw.efSearch = v
        elif self.kind == "ivf":
            self.idx.nprobe = v
        elif self.kind.startswith("usearch"):
            self.ef = v

    def search(self, q: np.ndarray, k: int):
        if self.kind.startswith("usearch"):
            m = self.idx.search(q, k, exact=False, threads=self.threads)
            if q.shape[0] == 1:
                return m.distances[None, :], m.keys[None, :].astype(np.int64)
            return m.distances, m.keys.astype(np.int64)
        return self.idx.search(q, k)


def recall_at_k(truth: np.ndarray, got: np.ndarray, k: int) -> float:
    r = 0.0
    for t, g in zip(truth, got):
        r += len(set(t[:k]) & set(g[:k])) / k
    return r / len(truth)


def cmd_build(args):
    x = load_vectors(args.emb)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res = []
    for kind in args.kinds:
        out = f"{args.out}.{kind}"
        r = build(kind, x, out, args.M, args.efc, args.nlist, args.threads)
        r["path"] = out
        print(json.dumps(r), flush=True)
        res.append(r)
    with open(f"{args.out}.build.json", "w") as f:
        json.dump(res, f, indent=1)


def cmd_bench(args):
    """Latency + recall@k vs exact for each index, across search-time params."""
    x = load_vectors(args.emb)
    rng = np.random.default_rng(0)
    qi = rng.choice(len(x), args.nq, replace=False)
    q = x[qi] + rng.normal(0, 0.02, size=(args.nq, x.shape[1])).astype(np.float32)  # perturbed corpus vectors as queries
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if args.queries:
        q = np.load(args.queries).astype(np.float32)[: args.nq]
    import faiss
    flat = faiss.IndexFlatIP(x.shape[1]); flat.add(x)
    t0 = time.perf_counter(); _, truth = flat.search(q, args.k); flat_batch = (time.perf_counter() - t0) / len(q) * 1000
    lat = []
    for i in range(min(200, len(q))):
        t = time.perf_counter(); flat.search(q[i:i + 1], args.k); lat.append((time.perf_counter() - t) * 1000)
    results = [{"kind": "flat", "param": 0, "recall@%d" % args.k: 1.0, "p50_ms": float(np.percentile(lat, 50)),
                "p99_ms": float(np.percentile(lat, 99)), "batch_ms_per_q": flat_batch, "n": len(x)}]
    print(json.dumps(results[0]), flush=True)
    for kind in args.kinds:
        path = f"{args.index}.{kind}"
        if not os.path.exists(path):
            continue
        s = Searcher(kind, path, threads=args.threads)
        params = args.params if kind != "flat" else [0]
        for p in params:
            s.set_param(p)
            s.search(q[:8], args.k)  # warm
            t0 = time.perf_counter(); _, got = s.search(q, args.k); batch = (time.perf_counter() - t0) / len(q) * 1000
            lat = []
            for i in range(min(300, len(q))):
                t = time.perf_counter(); s.search(q[i:i + 1], args.k); lat.append((time.perf_counter() - t) * 1000)
            r = {"kind": kind, "param": p, "recall@%d" % args.k: recall_at_k(truth, got, args.k),
                 "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)), "batch_ms_per_q": batch, "n": len(x)}
            print(json.dumps(r), flush=True)
            results.append(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--emb", default="data/index/emb")
    b.add_argument("--out", default="data/index/idx")
    b.add_argument("--kinds", nargs="+", default=["flat", "hnsw"])
    b.add_argument("--M", type=int, default=32)
    b.add_argument("--efc", type=int, default=200)
    b.add_argument("--nlist", type=int, default=0)
    b.add_argument("--threads", type=int, default=0)
    b.set_defaults(fn=cmd_build)
    e = sub.add_parser("bench")
    e.add_argument("--emb", default="data/index/emb")
    e.add_argument("--index", default="data/index/idx")
    e.add_argument("--kinds", nargs="+", default=["hnsw"])
    e.add_argument("--params", nargs="+", type=int, default=[16, 32, 64, 128])
    e.add_argument("--k", type=int, default=10)
    e.add_argument("--nq", type=int, default=1000)
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--queries", default="")
    e.add_argument("--out", default="bench/results/index_bench.json")
    e.set_defaults(fn=cmd_bench)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
