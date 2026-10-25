"""Wikipedia ANN sweep against exact ground truth (bench/ann_truth.py): recall and latency per index setting.

  uv run python bench/wiki_ann.py --index data/wiki/index/wiki.ivf --nprobe 16 32 64 128 --refine 50 100 200

Recall is measured the way search.chat consumes results: top-8 articles (best passage per article), and top-10
passages, both against brute force over the same (non-tombstoned) vectors.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

P = "data/wiki/index/wiki"


def top_articles(ids, scores, doc, k):
    out, seen = [], set()
    for i in ids[np.argsort(-scores)]:
        d = int(doc[i])
        if d not in seen:
            seen.add(d); out.append(d)
            if len(out) == k:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=P + ".ivf")
    ap.add_argument("--truth", default="data/eval/nq")
    ap.add_argument("--nprobe", type=int, nargs="+", default=[64])
    ap.add_argument("--refine", type=int, nargs="+", default=[100])
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--hnsw-quantizer", type=int, default=0, help="replace the flat coarse quantizer by HNSW(M)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.threads)

    q = np.load(a.truth + ".qv.npy")[:a.nq].astype(np.float32)
    n = int(np.load(a.truth + ".wiki.n.npy")[0])
    tid, tsc = np.load(a.truth + ".wiki.ids.npy")[:a.nq], np.load(a.truth + ".wiki.scores.npy")[:a.nq]
    vecs = np.memmap(P + ".f16", dtype=np.float16, mode="r", shape=(n, 384))
    doc = np.memmap(P + ".doc.i32", dtype=np.int32, mode="r", shape=(n,))
    dead = np.fromfile(P + ".dead.u8", np.uint8)[:n]
    truth_art = [top_articles(tid[j], tsc[j], doc, 8) for j in range(a.nq)]

    t0 = time.perf_counter()
    index = faiss.read_index(a.index)
    print(f"{a.index}: {index.ntotal:,} vectors, nlist {index.nlist}, load {time.perf_counter() - t0:.1f}s", flush=True)
    if a.hnsw_quantizer:
        cent = index.quantizer.reconstruct_n(0, index.nlist)
        hq = faiss.IndexHNSWFlat(index.d, a.hnsw_quantizer, faiss.METRIC_INNER_PRODUCT)
        hq.hnsw.efConstruction = 200
        hq.add(cent)
        index.quantizer = hq; index.own_fields = False; a._keep = hq
    res = []
    for npb in a.nprobe:
        index.nprobe = npb
        if a.hnsw_quantizer:
            a._keep.hnsw.efSearch = max(64, 2 * npb)
        for rf in a.refine:
            lat, r10, r8 = [], [], []
            for j in range(a.nq):
                t = time.perf_counter()
                _, I = index.search(q[j:j + 1], rf)
                ids = I[0][(I[0] >= 0) & (I[0] < n)]
                ids = ids[dead[ids] == 0] if len(dead) else ids
                ids.sort()
                sc = vecs[ids].astype(np.float32) @ q[j]
                lat.append((time.perf_counter() - t) * 1e3)
                top10 = ids[np.argsort(-sc)[:10]]
                r10.append(len(set(top10.tolist()) & set(tid[j][:10].tolist())) / 10)
                arts = top_articles(ids, sc, doc, 8)
                r8.append(len(set(arts) & set(truth_art[j])) / max(1, len(truth_art[j])))
            row = {"tag": a.tag, "nprobe": npb, "refine": rf, "hnswq": a.hnsw_quantizer, "threads": a.threads,
                   "recall10": round(float(np.mean(r10)), 4), "art_recall8": round(float(np.mean(r8)), 4),
                   "p50_ms": round(float(np.percentile(lat, 50)), 3), "p95_ms": round(float(np.percentile(lat, 95)), 3)}
            print(json.dumps(row), flush=True)
            res.append(row)
    with open("bench/results/wiki_ann.jsonl", "a") as f:
        for r in res:
            f.write(json.dumps({**r, "index": a.index}) + "\n")


if __name__ == "__main__":
    main()
