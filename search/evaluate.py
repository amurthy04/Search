"""Accuracy evaluation.

1) BEIR-style public benchmark (default: scifact, nfcorpus) -> nDCG@10, Recall@100, to sanity-check the
   off-the-shelf embedding model + our chunking/index against published MTEB numbers.
2) Self-eval on the crawled corpus with pseudo-queries (title / random sentence from a held-out page):
   is the source page in the top-k?  -> MRR@10, R@1, R@10.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import random
import time
import urllib.request
import zipfile

import numpy as np

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def load_beir(name: str, root: str = "data/beir"):
    d = os.path.join(root, name)
    if not os.path.exists(d):
        os.makedirs(root, exist_ok=True)
        z = os.path.join(root, name + ".zip")
        urllib.request.urlretrieve(BEIR_URL.format(name=name), z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    corpus = {}
    with open(os.path.join(d, "corpus.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            corpus[r["_id"]] = (r.get("title") or "", r.get("text") or "")
    queries = {}
    with open(os.path.join(d, "queries.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]
    qrels = collections.defaultdict(dict)
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, doc, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                qrels[q][doc] = int(s)
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def ndcg_at_k(ranked: list[str], rel: dict, k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def cmd_beir(args):
    if len(args.models) > 1:
        # one subprocess per model: faiss (own OpenMP) + transformers' threaded weight loading segfault in one process
        import subprocess, sys
        for m in args.models:
            subprocess.run([sys.executable, "-m", "search.evaluate", "beir", "--models", m, "--datasets", *args.datasets,
                            "--backend", args.backend, "--device", args.device, "--max-seq", str(args.max_seq),
                            "--batch", str(args.batch), "--out", args.out])
        return
    from .embed import make_embedder
    results = {}
    for model in args.models:
        emb = make_embedder(args.backend, model, device=args.device, fp16=True, max_seq=args.max_seq, batch_size=args.batch)
        import faiss
        faiss.omp_set_num_threads(1)  # multi-threaded faiss segfaults once torch has initialised its own OpenMP runtime
        for name in args.datasets:
            corpus, queries, qrels = load_beir(name)
            ids = list(corpus)
            texts = [(f"{t}\n{x}" if t else x) for t, x in (corpus[i] for i in ids)]
            t0 = time.perf_counter()
            X = emb.encode(texts, show=False).astype(np.float32)
            t_emb = time.perf_counter() - t0
            idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)
            qids = list(queries)
            t0 = time.perf_counter()
            Q = emb.encode([queries[q] for q in qids], is_query=True).astype(np.float32)
            t_q = time.perf_counter() - t0
            _, I = idx.search(Q, 100)
            nd, r100 = [], []
            for qi, q in enumerate(qids):
                ranked = [ids[j] for j in I[qi]]
                nd.append(ndcg_at_k(ranked, qrels[q], 10))
                r100.append(len(set(ranked) & set(qrels[q])) / len(qrels[q]))
            r = {"model": model, "dataset": name, "ndcg@10": float(np.mean(nd)), "recall@100": float(np.mean(r100)),
                 "n_docs": len(ids), "n_queries": len(qids), "docs_per_s": len(ids) / t_emb, "query_ms": t_q / len(qids) * 1e3}
            print(json.dumps(r), flush=True)
            results[f"{model}|{name}"] = r
        del emb
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        for r in results.values():
            f.write(json.dumps(r) + "\n")


def pseudo_queries(shards: str, n: int, seed: int = 0, kind: str = "title"):
    """Sample n pages; build (query, url) pairs."""
    from .corpus import iter_records
    rng = random.Random(seed)
    pairs = []
    recs = []
    for i, rec in enumerate(iter_records(shards)):
        if len(rec.get("text", "")) < 800 or not rec.get("title") or (rec.get("lang") or "en")[:2] != "en":
            continue
        # reservoir sample
        if len(recs) < n:
            recs.append(rec)
        else:
            j = rng.randrange(i + 1)
            if j < n:
                recs[j] = rec
    for rec in recs:
        if kind == "title":
            q = rec["title"]
        else:
            sents = [s.strip() for s in rec["text"].replace("\n", " ").split(". ") if 60 < len(s.strip()) < 200]
            if not sents:
                continue
            q = rng.choice(sents)
        pairs.append((q, rec["url"]))
    return pairs


def cmd_self(args):
    from .query import Engine
    eng = Engine(args.model, args.emb, args.index, args.kind, args.backend, args.device, args.ef, 1)
    out = {}
    for kind in ("title", "sentence"):
        pairs = pseudo_queries(args.shards, args.n, kind=kind)
        mrr = r1 = r10 = 0.0
        lat = []
        for q, url in pairs:
            res = eng.search(q, 10, snippets=False)
            lat.append(res["t_ms"]["total"])
            urls = [h["url"] for h in res["hits"]]
            if url in urls:
                rank = urls.index(url) + 1
                mrr += 1 / rank; r10 += 1; r1 += rank == 1
        n = max(1, len(pairs))
        out[kind] = {"n": len(pairs), "mrr@10": mrr / n, "r@1": r1 / n, "r@10": r10 / n,
                     "lat_p50_ms": float(np.percentile(lat, 50)), "lat_p99_ms": float(np.percentile(lat, 99))}
        print(kind, json.dumps(out[kind]), flush=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "index": args.index, **out}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("beir")
    b.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    b.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    b.add_argument("--backend", default="st"); b.add_argument("--device", default="mps")
    b.add_argument("--max-seq", type=int, default=512); b.add_argument("--batch", type=int, default=64)
    b.add_argument("--out", default="bench/results/beir.jsonl")
    b.set_defaults(fn=cmd_beir)
    s = sub.add_parser("self")
    s.add_argument("--shards", default="data/crawl/*.jsonl.zst")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--model", default="BAAI/bge-small-en-v1.5"); s.add_argument("--backend", default="st"); s.add_argument("--device", default="mps")
    s.add_argument("--emb", default="data/index/emb"); s.add_argument("--index", default="data/index/idx.hnsw"); s.add_argument("--kind", default="hnsw")
    s.add_argument("--ef", type=int, default=64)
    s.add_argument("--out", default="bench/results/self_eval.json")
    s.set_defaults(fn=cmd_self)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
