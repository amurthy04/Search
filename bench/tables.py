"""Render markdown tables from bench/results/*.json(l) for RESULTS.md."""
import glob, json, os, sys

R = "bench/results"


def beir():
    rows = [json.loads(l) for l in open(f"{R}/beir.jsonl")] if os.path.exists(f"{R}/beir.jsonl") else []
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["dataset"]] = r
    out = ["| model | SciFact nDCG@10 | NFCorpus nDCG@10 | SciFact R@100 | NFCorpus R@100 |", "|---|---|---|---|---|"]
    for m, d in sorted(by.items(), key=lambda kv: -kv[1].get("scifact", {}).get("ndcg@10", 0)):
        g = lambda ds, k: f"{d[ds][k]:.3f}" if ds in d else ""
        out.append(f"| {m} | {g('scifact','ndcg@10')} | {g('nfcorpus','ndcg@10')} | {g('scifact','recall@100')} | {g('nfcorpus','recall@100')} |")
    return "\n".join(out)


def index():
    out = ["| index | param | recall@10 | p50 ms | p99 ms | batch ms/q |", "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{R}/index_bench_*.json")):
        for r in json.load(open(f)):
            k = [x for x in r if x.startswith("recall@")][0]
            out.append(f"| {r['kind']} | {r['param']} | {r[k]:.3f} | {r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['batch_ms_per_q']:.3f} |")
    return "\n".join(out)


def e2e():
    out = ["| tag | model | backend | passages | embed p50 | search p50 | total p50 | total p95 | total p99 |", "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{R}/e2e_*.json")):
        s = json.load(open(f)); L = s["latency_ms"]
        out.append(f"| {s['tag']} | {s['model'].split('/')[-1]} | {s['backend']}/{s['device']} | {s['n_passages']} | {L['embed']['p50']:.1f} | {L['search']['p50']:.2f} | {L['total']['p50']:.1f} | {L['total']['p95']:.1f} | {L['total']['p99']:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["beir", "index", "e2e"]:
        print(f"### {name}\n{globals()[name]()}\n")
