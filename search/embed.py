"""Embed passages with an off-the-shelf model; benchmark backends.

Backends:
  st   - sentence-transformers on torch (mps/cpu), fp16 optional
  mlx  - MLX (Apple silicon) via mlx_embeddings (optional)
Output: <out>.npy (float16 or float32 [N, D]), <out>.meta.jsonl.zst (doc_id, chunk_id, url, title)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import orjson
import zstandard as zstd

from .corpus import iter_passages

# model -> (query_prefix, doc_prefix, trust_remote_code)
MODELS = {
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", "", False),
    "BAAI/bge-base-en-v1.5": ("Represent this sentence for searching relevant passages: ", "", False),
    "BAAI/bge-m3": ("", "", False),
    "intfloat/e5-base-v2": ("query: ", "passage: ", False),
    "intfloat/multilingual-e5-small": ("query: ", "passage: ", False),
    "intfloat/multilingual-e5-base": ("query: ", "passage: ", False),
    "Snowflake/snowflake-arctic-embed-m-v1.5": ("Represent this sentence for searching relevant passages: ", "", False),
    "Snowflake/snowflake-arctic-embed-s": ("Represent this sentence for searching relevant passages: ", "", False),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: ", True),
    "Alibaba-NLP/gte-base-en-v1.5": ("", "", True),
    "Alibaba-NLP/gte-modernbert-base": ("", "", False),
    "Qwen/Qwen3-Embedding-0.6B": ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", "", False),
    "sentence-transformers/all-MiniLM-L6-v2": ("", "", False),
    "sentence-transformers/static-retrieval-mrl-en-v1": ("", "", False),
    "mixedbread-ai/mxbai-embed-large-v1": ("Represent this sentence for searching relevant passages: ", "", False),
}


def prefixes(model_name: str):
    return MODELS.get(model_name, ("", "", False))


# pooling used by each model family (needed for the MLX backend, which exposes raw hidden states)
POOLING = {"BAAI/bge": "cls", "Snowflake/snowflake-arctic": "cls", "intfloat/e5": "mean", "intfloat/multilingual-e5": "mean",
           "sentence-transformers/all-MiniLM": "mean", "Alibaba-NLP/gte-modernbert": "cls", "nomic-ai": "mean", "mixedbread-ai": "cls"}


def pooling_for(model_name: str) -> str:
    for k, v in POOLING.items():
        if model_name.startswith(k):
            return v
    return "mean"


class STEmbedder:
    def __init__(self, model_name: str, device: str = "mps", fp16: bool = True, max_seq: int = 256, batch_size: int = 128):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.qp, self.dp, trc = prefixes(model_name)
        kw = {"trust_remote_code": trc}
        if fp16:
            kw["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = SentenceTransformer(model_name, device=device, **kw)
        try:
            self.model.max_seq_length = max_seq
        except AttributeError:
            pass  # static embeddings have no sequence limit
        self.batch_size = batch_size
        self.device = device
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], is_query=False, show=False) -> np.ndarray:
        pre = self.qp if is_query else self.dp
        if pre:
            texts = [pre + t for t in texts]
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show)


class MLXEmbedder:
    """MLX backend: runs the HF transformer with mlx (bert/xlm-roberta family) with mean/CLS pooling."""

    def __init__(self, model_name: str, max_seq: int = 256, batch_size: int = 128, pooling: str = ""):
        import mlx.core as mx
        from mlx_embeddings import load
        self.mx = mx
        self.model, self.tokenizer = load(model_name)
        self.model_name = model_name
        self.qp, self.dp, _ = prefixes(model_name)
        self.max_seq = max_seq
        self.batch_size = batch_size
        self.pooling = pooling or pooling_for(model_name)
        self.dim = None

    def encode(self, texts, is_query=False, show=False):
        mx = self.mx
        pre = self.qp if is_query else self.dp
        if pre:
            texts = [pre + t for t in texts]
        out = []
        order = np.argsort([-len(t) for t in texts])
        for i in range(0, len(texts), self.batch_size):
            idx = order[i:i + self.batch_size]
            batch = [texts[j] for j in idx]
            enc = self.tokenizer._tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=self.max_seq)
            ids = mx.array(enc["input_ids"]); mask = mx.array(enc["attention_mask"])
            res = self.model(ids, attention_mask=mask)
            hs = res.last_hidden_state
            if self.pooling == "cls":
                emb = hs[:, 0, :]
            else:
                m = mask[:, :, None].astype(hs.dtype)
                emb = (hs * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1e-9)
            emb = emb / mx.linalg.norm(emb, axis=1, keepdims=True)
            mx.eval(emb)
            arr = np.array(emb.astype(mx.float32))
            out.append((idx, arr))
        dim = out[0][1].shape[1]
        self.dim = dim
        res = np.zeros((len(texts), dim), dtype=np.float32)
        for idx, arr in out:
            res[idx] = arr
        return res


class CoreMLEmbedder:
    """CoreML (Apple GPU / ANE) backend. Fixed-shape fp16 graph exported from search.minibert via bench/coreml_convert.py.

    Batches are padded to (B, S); the fixed graph makes the per-batch cost constant, so passages are sorted by
    length upstream and the last partial batch is padded. ~2.2x the throughput of torch/MPS for bge-small.
    """

    def __init__(self, model_name: str, max_seq: int = 256, batch_size: int = 32, compute: str = "gpu", **_):
        import coremltools as ct
        from transformers import AutoTokenizer
        from .minibert import MiniBert
        import torch
        self.model_name = model_name
        self.qp, self.dp, _ = prefixes(model_name)
        self.B, self.S = batch_size, max_seq
        path = f"data/index/coreml_{model_name.split('/')[-1]}_b{batch_size}_s{max_seq}.mlpackage"
        if not os.path.exists(path):
            m = MiniBert.from_hf(model_name, pooling_for(model_name), fixed=(batch_size, max_seq))
            ids = torch.randint(100, 1000, (batch_size, max_seq)); mask = torch.ones(batch_size, max_seq, dtype=torch.long)
            traced = torch.jit.trace(m, (ids, mask))
            ml = ct.convert(traced, inputs=[ct.TensorType(name="input_ids", shape=(batch_size, max_seq), dtype=np.int32),
                                            ct.TensorType(name="attention_mask", shape=(batch_size, max_seq), dtype=np.int32)],
                            outputs=[ct.TensorType(name="emb")], compute_precision=ct.precision.FLOAT16,
                            minimum_deployment_target=ct.target.macOS15, convert_to="mlprogram")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ml.save(path)
        cu = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}[compute]
        self.model = ct.models.MLModel(path, compute_units=cu)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.dim = None

    def _batches(self, texts):
        """Tokenize batch i+1 on a worker thread while batch i runs on the GPU."""
        from concurrent.futures import ThreadPoolExecutor
        B, S = self.B, self.S

        def tok(chunk):
            enc = self.tok(chunk, return_tensors="np", padding="max_length", truncation=True, max_length=S)
            ids = enc["input_ids"].astype(np.int32); mask = enc["attention_mask"].astype(np.int32)
            if len(chunk) < B:
                pad = B - len(chunk)
                ids = np.concatenate([ids, np.zeros((pad, S), np.int32)]); mask = np.concatenate([mask, np.zeros((pad, S), np.int32)])
                mask[-pad:, 0] = 1
            return len(chunk), ids, mask

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(tok, texts[i:i + B]) for i in range(0, len(texts), B)]
            for f in futs:
                yield f.result()

    def encode(self, texts, is_query=False, show=False):
        pre = self.qp if is_query else self.dp
        if pre:
            texts = [pre + t for t in texts]
        out = []
        for n, ids, mask in self._batches(texts):
            r = self.model.predict({"input_ids": ids, "attention_mask": mask})["emb"]
            out.append(r[:n])
        res = np.concatenate(out).astype(np.float32)
        self.dim = res.shape[1]
        return res


def make_embedder(backend: str, model: str, **kw):
    if backend == "st":
        return STEmbedder(model, **kw)
    if backend == "mlx":
        kw.pop("device", None); kw.pop("fp16", None)
        return MLXEmbedder(model, **kw)
    if backend in ("coreml", "coreml_ane", "coreml_all"):
        kw.pop("device", None); kw.pop("fp16", None)
        return CoreMLEmbedder(model, compute={"coreml": "gpu", "coreml_ane": "ane", "coreml_all": "all"}[backend], **kw)
    raise ValueError(backend)


def cmd_bench(args):
    """Throughput benchmark on synthetic passages of ~args.words words."""
    import random
    rng = random.Random(0)
    vocab = ["the", "search", "engine", "web", "crawl", "latency", "vector", "embedding", "index", "query",
             "model", "fast", "python", "network", "page", "link", "data", "open", "source", "benchmark"] * 5
    texts = [" ".join(rng.choice(vocab) for _ in range(args.words)) for _ in range(args.n)]
    results = []
    for model in args.models:
        for bs in args.batch_sizes:
            try:
                t0 = time.time()
                emb = make_embedder(args.backend, model, device=args.device, fp16=not args.fp32, max_seq=args.max_seq, batch_size=bs)
                load_s = time.time() - t0
                emb.encode(texts[:bs * 2])  # warmup
                t0 = time.perf_counter()
                vecs = emb.encode(texts)
                dt = time.perf_counter() - t0
                r = {"model": model, "backend": args.backend, "batch": bs, "max_seq": args.max_seq, "n": args.n,
                     "load_s": round(load_s, 1), "docs_per_s": round(args.n / dt, 1), "dim": int(vecs.shape[1])}
                print(json.dumps(r), flush=True)
                results.append(r)
                del emb
                import gc; gc.collect()
                try:
                    import torch; torch.mps.empty_cache()
                except Exception:
                    pass
            except Exception as e:
                print(json.dumps({"model": model, "batch": bs, "error": repr(e)[:300]}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


def finalize(out: str, dim: int) -> np.ndarray:
    """Turn the incrementally written .f16.bin into .npy (also usable to recover a crashed run)."""
    raw = np.fromfile(out + ".f16.bin", dtype=np.float16)
    vecs = raw.reshape(-1, dim)
    np.save(out + ".npy", vecs)
    return vecs


def cmd_run(args):
    """Embed the crawl corpus to disk."""
    t_all = time.perf_counter()
    emb = make_embedder(args.backend, args.model, device=args.device, fp16=not args.fp32, max_seq=args.max_seq, batch_size=args.batch_size)
    print(f"model loaded in {time.perf_counter()-t_all:.1f}s dim={emb.dim}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta_fh = open(args.out + ".meta.jsonl.zst", "wb")
    meta_w = zstd.ZstdCompressor(level=3).stream_writer(meta_fh)
    pass_fh = open(args.out + ".passages.bin", "wb")
    vec_fh = open(args.out + ".f16.bin", "wb")  # vectors appended per flush; a crash keeps everything so far
    offsets = [0]
    vec_chunks = []
    buf_texts, buf_meta = [], []
    n = 0
    t_read = t_embed = 0.0
    t0 = time.perf_counter()
    dim_holder = [0]

    def flush():
        nonlocal t_embed
        if not buf_texts:
            return
        t = time.perf_counter()
        # sort by length within the flush block to minimize padding
        order = np.argsort([len(x) for x in buf_texts])
        texts = [buf_texts[i] for i in order]
        vecs = emb.encode(texts)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        vecs = vecs[inv]
        v16 = np.ascontiguousarray(vecs.astype(np.float16))
        vec_fh.write(v16.tobytes()); dim_holder[0] = v16.shape[1]
        for m, txt in zip(buf_meta, buf_texts):
            meta_w.write(orjson.dumps(m) + b"\n")
            b = txt.encode("utf-8")
            pass_fh.write(b)
            offsets.append(offsets[-1] + len(b))
        t_embed += time.perf_counter() - t
        buf_texts.clear(); buf_meta.clear()

    tr = time.perf_counter()
    for doc_id, ci, url, title, passage in iter_passages(args.shards, args.words, args.overlap, args.max_chunks):
        t_read += time.perf_counter() - tr
        buf_texts.append(passage)
        buf_meta.append({"d": doc_id, "c": ci, "u": url, "t": title})
        n += 1
        if len(buf_texts) >= args.flush:
            flush()
            if n % (args.flush * 10) == 0:
                el = time.perf_counter() - t0
                print(f"{n} passages {n/el:.0f}/s (embed {t_embed:.0f}s, read {t_read:.0f}s)", flush=True)
        if args.limit and n >= args.limit:
            break
        tr = time.perf_counter()
    flush()
    meta_w.close(); meta_fh.close(); pass_fh.close(); vec_fh.close()
    np.save(args.out + ".offsets.npy", np.asarray(offsets, dtype=np.int64))
    vecs = finalize(args.out, dim_holder[0] or emb.dim or 1)
    el = time.perf_counter() - t0
    summary = {"model": args.model, "backend": args.backend, "passages": n, "dim": int(vecs.shape[1]), "elapsed_s": el,
               "passages_per_s": n / el, "embed_s": t_embed, "read_s": t_read, "max_seq": args.max_seq, "batch": args.batch_size}
    print(json.dumps(summary, indent=1))
    with open(args.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--models", nargs="+", default=["BAAI/bge-small-en-v1.5"])
    b.add_argument("--backend", default="st")
    b.add_argument("--device", default="mps")
    b.add_argument("--fp32", action="store_true")
    b.add_argument("--max-seq", type=int, default=256)
    b.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    b.add_argument("--n", type=int, default=2048)
    b.add_argument("--words", type=int, default=180)
    b.add_argument("--out", default="bench/results/embed_bench.jsonl")
    b.set_defaults(fn=cmd_bench)
    r = sub.add_parser("run")
    r.add_argument("--shards", default="data/crawl/*.jsonl.zst")
    r.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    r.add_argument("--backend", default="st")
    r.add_argument("--device", default="mps")
    r.add_argument("--fp32", action="store_true")
    r.add_argument("--max-seq", type=int, default=256)
    r.add_argument("--batch-size", type=int, default=128)
    r.add_argument("--flush", type=int, default=4096)
    r.add_argument("--words", type=int, default=180)
    r.add_argument("--overlap", type=int, default=30)
    r.add_argument("--max-chunks", type=int, default=6)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default="data/index/emb")
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
