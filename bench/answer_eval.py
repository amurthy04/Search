"""End-to-end answer accuracy of the chat pipeline: NQ-open questions -> search.chat.retrieve() -> Qwen3-0.6B
(greedy, no thinking) -> does the answer contain a gold answer (normalized)?

  uv run python bench/answer_eval.py --n 500 --modes ivf none          # new index, and the LLM without retrieval
  uv run python bench/answer_eval.py --n 500 --modes legacy --ivf /nonexistent

Writes bench/results/answers_<mode>.jsonl (question, gold, context sources, answer, hit).
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import search.chat as chat  # noqa: E402
import numpy as np  # noqa: E402
from retrieval import _Stub, has_answer  # noqa: E402


def ask(llm, sid, user, max_tokens):
    out = llm.submit(sid, user, False, max_tokens, threading.Event(), greedy=True)
    text = []
    while (ev := out.get()) is not None:
        if "t" in ev and isinstance(ev["t"], str):
            text.append(ev["t"])
    llm.reset(sid)
    return "".join(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qs", default="data/eval/nq_open_dev.jsonl")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["ivf"], help="ivf | legacy | none (no retrieval) | nolink")
    ap.add_argument("--ivf", default="data/ivf"); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-0.6B-mlx-4bit"))
    ap.add_argument("--variants", default='[{}]', help='JSON list of {"urls", "k_ctx", "thr", "system"} overrides')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.qs)][a.offset:a.offset + a.n]
    live = chat.LiveIndex(ivf_dir=a.ivf)
    while getattr(live.tiers, "busy", False) or getattr(live.tiers, "sealing", False):
        time.sleep(0.5)
    llm = chat.LLM(a.model, threading.Event())
    link = live.cards.link
    system0 = chat.SYSTEM
    for mode, var in [(m, v) for m in a.modes for v in json.loads(a.variants)]:
        live.cards.link = (lambda q: []) if mode == "nolink" else link
        chat.SYSTEM = var.get("system", system0)
        kw = {{"urls": "source_urls"}.get(k, k): var[k] for k in ("urls", "k_ctx", "thr") if k in var}
        hits, ctx_hits, out = [], [], []
        t0 = time.time()
        for i, r in enumerate(rows):
            if mode == "none":
                ctx, srcs = "", []
            else:
                ctx, srcs, _, _, _, _ = chat.retrieve(live, None, _Stub(), r["q"], None, lambda e: None, **kw)
            ans = ask(llm, f"eval{i}", f"{ctx}{r['q']}", a.max_tokens)
            hit = has_answer(ans, r["a"])
            hits.append(hit); ctx_hits.append(has_answer(ctx, r["a"]) if ctx else False)
            out.append({"q": r["q"], "a": r["a"], "answer": ans, "hit": hit, "ctx_hit": ctx_hits[-1],
                        "sources": [(s["kind"], s["title"], s["score"]) for s in srcs]})
        os.makedirs("bench/results", exist_ok=True)
        tag = mode + ("_" + "_".join(f"{k}{v if k != 'system' else len(v)}" for k, v in var.items()) if var else "")
        with open(f"bench/results/answers_{tag}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        print(json.dumps({"mode": mode, **{k: (v if k != "system" else v[:40]) for k, v in var.items()}, "n": len(rows), "offset": a.offset, "answer_acc": round(float(np.mean(hits)), 4),
                          "ctx_answer_recall": round(float(np.mean(ctx_hits)), 4),
                          "acc_when_ctx_has_answer": round(float(np.mean([h for h, c in zip(hits, ctx_hits) if c] or [0])), 4),
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
