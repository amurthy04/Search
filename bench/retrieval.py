"""Retrieval benchmark for the answer engine: latency of every stage of search.chat.retrieve() and whether the
gold answer reaches the LLM's context, on NQ-open questions (fresh queries, each seen once).

  uv run python bench/retrieval.py --n 500 --tag base
  uv run python bench/retrieval.py --n 500 --offset 500 --tag base   # a disjoint, never-seen set of questions

Writes bench/results/retrieval_<tag>.json (summary + per-question rows).
"""
import argparse
import json
import os
import re
import string
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import search.chat as chat  # noqa: E402  (sets HF_HUB_OFFLINE / OMP_NUM_THREADS and chdir before faiss/CoreML load)
import numpy as np  # noqa: E402


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def has_answer(text: str, answers: list[str]) -> bool:
    t = " " + norm(text) + " "
    return any((" " + norm(a) + " ") in t for a in answers if norm(a))


class _Stub:
    def pause_crawler(self, *_):
        pass


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 3) if len(xs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", default="data/eval/nq_open_dev.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--emb", default="data/index/emb_all")
    ap.add_argument("--index", default="data/index/idx_all.hnsw")
    ap.add_argument("--ivf", default="data/ivf", help="shared IVF dir (a missing dir = the legacy HNSW + wiki .ivf path)")
    ap.add_argument("--nprobe", type=int, default=0); ap.add_argument("--wiki-refine", type=int, default=0)
    ap.add_argument("--web-refine", type=int, default=0)
    ap.add_argument("--sweep", default="", help='JSON list of {"nprobe", "wiki_refine", "web_refine"} settings')
    ap.add_argument("--lat-qs", default="data/eval/nq_open_train.jsonl")
    ap.add_argument("--lat-offset", type=int, default=10000); ap.add_argument("--lat-n", type=int, default=300)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.qs)][a.offset:a.offset + a.n]
    t0 = time.perf_counter()
    live = chat.LiveIndex(a.emb, a.index, ivf_dir=a.ivf)
    print(f"loaded in {time.perf_counter() - t0:.1f}s: web {live.base_n:,} + wiki {live.wiki.n:,}", flush=True)
    if a.nprobe and live.ivf is not None:
        live.ivf.nprobe = a.nprobe
    if a.wiki_refine:
        live.wiki.refine = a.wiki_refine
    if a.web_refine:
        live.web_refine = a.web_refine

    # time the parts retrieve() doesn't report itself
    acc = {"link_ms": 0.0, "passage_ms": 0.0, "card_ms": 0.0}

    def timed(fn, key):
        def w(*x, **kw):
            t = time.perf_counter()
            try:
                return fn(*x, **kw)
            finally:
                acc[key] += (time.perf_counter() - t) * 1e3
        return w
    live.cards.link = timed(live.cards.link, "link_ms")
    live.cards.card = timed(live.cards.card, "card_ms")
    live.cards.get = timed(live.cards.get, "card_ms")
    live.passage = timed(live.passage, "passage_ms")
    live.wiki.passage = timed(live.wiki.passage, "passage_ms")
    t0 = time.time()  # startup work (live passages being indexed) would otherwise be measured as query latency
    while getattr(live.tiers, "busy", False) or getattr(live.tiers, "sealing", False):
        time.sleep(0.5)
    print(f"background indexing done {time.time() - t0:.1f}s after load", flush=True)
    for q in ("warm up one", "warm up two", "warm up three"):  # first searches touch cold code paths
        chat.retrieve(live, None, _Stub(), q, None, lambda e: None)

    if a.sweep:  # several settings in one process: accuracy on --qs rows, latency on fresh --lat-qs rows each
        lat_rows = [json.loads(l) for l in open(a.lat_qs)]
        for i, cfg in enumerate(json.loads(a.sweep)):
            if live.ivf is not None:
                live.ivf.nprobe = cfg.get("nprobe", live.ivf.nprobe)
            live.wiki.refine = cfg.get("wiki_refine", live.wiki.refine)
            live.web_refine = cfg.get("web_refine", live.web_refine)
            acc_out = run(live, rows, acc)
            lat_out = run(live, lat_rows[a.lat_offset + i * a.lat_n:a.lat_offset + (i + 1) * a.lat_n], acc)
            tot = [o["total_ms"] for o in lat_out]
            print(json.dumps({**cfg, "ctx_answer_recall": round(float(np.mean([o["ctx_hit"] for o in acc_out])), 4),
                              "fresh_p50": pct(tot, 50), "fresh_p95": pct(tot, 95), "fresh_p99": pct(tot, 99),
                              "search_p50": pct([o["search_ms"] for o in lat_out], 50)}), flush=True)
        return
    out = run(live, rows, acc)

    keys = ["total_ms", "embed_ms", "hnsw_ms", "wiki_ms", "search_ms", "link_ms", "passage_ms", "card_ms"]
    summ = {"tag": a.tag, "n": len(out), "offset": a.offset, "qs": a.qs,
            "nprobe": getattr(live.ivf, "nprobe", None), "wiki_refine": live.wiki.refine, "web_refine": getattr(live, "web_refine", None), "web_passages": live.base_n, "wiki_passages": live.wiki.n,
            "ctx_answer_recall": round(float(np.mean([o["ctx_hit"] for o in out])), 4),
            "mean_sources": round(float(np.mean([o["n_src"] for o in out])), 2)}
    for k in keys:
        xs = [o[k] for o in out if k in o]
        summ[k] = {"p50": pct(xs, 50), "p95": pct(xs, 95), "p99": pct(xs, 99), "mean": round(float(np.mean(xs)), 3)}
    print(json.dumps(summ, indent=1))
    os.makedirs("bench/results", exist_ok=True)
    json.dump({"summary": summ, "rows": out}, open(f"bench/results/retrieval_{a.tag}.json", "w"), indent=1)


def run(live, rows, acc):
    out = []
    for r in rows:
        for k in acc:
            acc[k] = 0.0
        t = time.perf_counter()
        ctx, sources, near, q, tm, _ = chat.retrieve(live, None, _Stub(), r["q"], None, lambda e: None)
        total = (time.perf_counter() - t) * 1e3
        out.append({"q": r["q"], "a": r["a"], "total_ms": total, **tm, **dict(acc),
                    "ctx_hit": has_answer(ctx, r["a"]), "n_src": len(sources),
                    "kinds": [s["kind"] for s in sources], "top": near[0]["score"] if near else 0.0})
    return out


if __name__ == "__main__":
    main()
