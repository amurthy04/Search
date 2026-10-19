# Search

A from-scratch web search stack tuned for latency on a single machine:

```
crawl (search/crawler.py)  ->  embed (search/embed.py)  ->  index (search/index.py)  ->  query (search/query.py)
   pages/s, per-stage ms        passages/s                  build s, recall@10, ms      embed/search/dedupe ms
```

Every stage records its own latency, and `bench/` holds the scripts and raw results of every
optimization step (what was tried, what moved, where returns diminished).

## Quick start

```bash
uv sync                                                         # Python 3.12 env
ulimit -n 65536
uv run python -m search.crawler --workers 8 --n-seeds 40000 --time-limit 600 --out data/crawl/pages
uv run python -m search.embed run --backend mlx --model BAAI/bge-small-en-v1.5 --out data/index/emb
uv run python -m search.index build --emb data/index/emb --out data/index/idx --kinds flat hnsw
uv run python -m search.query repl --index data/index/idx.hnsw
uv run python bench/e2e.py --tag mine --print                   # 30 queries, per-stage latency
```

## Components

### 1. Crawler
Multi-process asyncio crawler. Each worker owns the hosts whose `xxhash(site) % N == worker_id`, so URL
dedup, robots.txt caching and per-host politeness are process-local and no coordinator is needed; links
for foreign hosts are batched over `multiprocessing` queues. Per worker: `aiohttp` + `uvloop`, keep-alive
pool, a pure-asyncio `dnspython` resolver with positive/negative caching, `selectolax` (lexbor) parsing,
`ada-url` normalization, and a 64-bit hashed seen-set. Breadth-first across hosts with per-host caps keeps
the corpus diverse. Output is zstd JSONL with `{url, title, lang, text, t: {dns, connect, ttfb, download, parse}}`.

### 2. Embeddings
Off-the-shelf models via `sentence-transformers` (torch/MPS) or an MLX backend (Apple silicon). Pages are
split into ~180-word passages prefixed with the title; passages are sorted by length to minimize padding.
Vectors are stored float16 (`.npy`) with a sidecar passage store for snippets.

### 3. Index
FAISS flat (exact baseline), FAISS HNSW / HNSW+SQ8, IVF, and usearch HNSW (f32/f16/i8). `index bench`
reports recall@10 vs exact and p50/p99 latency per search-time parameter.

### 4. Query
Embed query -> ANN top-k·overfetch -> dedupe by document -> snippets. Each stage is timed and returned.
`query serve` exposes `GET /search?q=...&k=10`.

### Evaluation
- `evaluate beir`: nDCG@10 / Recall@100 on BEIR SciFact + NFCorpus, to check the off-the-shelf model against
  published MTEB numbers independent of our crawl.
- `evaluate self`: pseudo-queries (page title, random sentence) from held-out crawled pages -> MRR@10, R@1, R@10.
- `bench/e2e.py`: 30 realistic web queries, per-stage latency distribution, and the full result lists for manual review.

## Results
See [RESULTS.md](RESULTS.md) for the optimization log and current numbers.
