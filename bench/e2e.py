"""End-to-end query benchmark: run a fixed query set through the Engine, record per-stage latency.

Writes bench/results/e2e_<tag>.json with p50/p95/p99 per stage + every result list for manual review.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.query import Engine, fmt

QUERIES = [
    "how do transformers work in machine learning",
    "best way to learn rust programming",
    "symptoms of vitamin d deficiency",
    "how to make sourdough bread at home",
    "what causes inflation",
    "climate change effects on coral reefs",
    "python asyncio tutorial",
    "history of the roman empire",
    "how to train for a marathon",
    "electric car battery lifespan",
    "quantum computing explained simply",
    "how does the stock market work",
    "cheap flights to tokyo",
    "recipe for chicken tikka masala",
    "what is the capital of australia",
    "how to fix a leaking faucet",
    "benefits of meditation",
    "javascript promises vs async await",
    "who won the world cup 2022",
    "how do vaccines work",
    "best laptops for programming 2026",
    "how to start a small business",
    "what is a vector database",
    "difference between tcp and udp",
    "how to write a resume",
    "healthy breakfast ideas",
    "how does gps work",
    "learn spanish fast",
    "linux command to find large files",
    "why is the sky blue",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5"); ap.add_argument("--backend", default="st"); ap.add_argument("--device", default="mps")
    ap.add_argument("--emb", default="data/index/emb"); ap.add_argument("--index", default="data/index/idx.hnsw"); ap.add_argument("--kind", default="hnsw")
    ap.add_argument("--ef", type=int, default=64); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3); ap.add_argument("--tag", default="run"); ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    eng = Engine(a.model, a.emb, a.index, a.kind, a.backend, a.device, a.ef, a.threads)
    cold = time.perf_counter() - t0
    stages = {}
    results = []
    for r in range(a.repeat):
        for q in QUERIES:
            res = eng.search(q, a.k)
            for k, v in res["t_ms"].items():
                stages.setdefault(k, []).append(v)
            if r == 0:
                results.append(res)
                if a.print:
                    print(fmt(res)); print()
    summary = {"tag": a.tag, "model": a.model, "backend": a.backend, "device": a.device, "index": a.index, "ef": a.ef, "k": a.k,
               "n_passages": len(eng.urls), "cold_start_s": cold, "load": {"model_s": eng.t_model, "index_s": eng.t_index, "meta_s": eng.t_meta},
               "latency_ms": {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "mean": float(np.mean(v))} for k, v in stages.items()},
               "results": results}
    os.makedirs("bench/results", exist_ok=True)
    with open(f"bench/results/e2e_{a.tag}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
