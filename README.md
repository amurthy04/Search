# Search

A search engine and question-answering chat that run on one laptop (M5 Pro MacBook, 48 GB RAM). It crawls the web,
imports English Wikipedia and Wikidata from their public dumps, turns every chunk of text into a vector with a small
embedding model, and answers questions with a 0.6B-parameter language model that reads the top results. Answering a
question makes no calls to outside services.

## What's in the index

| source | documents | passages |
|---|---|---|
| Web crawl (Tranco top-1M seeds + 239 reference sites) | 675k pages | 2.41M |
| English Wikipedia + Wikidata (Oct 2026 dumps, updated hourly) | 13.4M | 42.2M (35.6M from articles, 6.6M Wikidata cards) |

A passage is a ~180-word chunk of a page, starting with the page title (and, for Wikipedia, the section name). Search
ranks passages, then keeps the best one per page.

## Speed

| | |
|---|---|
| web search, end to end | 1.9 ms median: 1.7 ms to embed the query, 0.2 ms for everything else |
| Wikipedia search | 1.9 ms median once the index is in memory, ~12 ms right after loading |
| chat, time to first token | ~100 ms |
| chat, generation | ~370 tokens/s |
| crawl | up to 321 pages/s |
| embedding passages | ~1,100/s on the GPU, ~330/s on the Neural Engine |

How each number was measured, and every optimization step tried, is in [RESULTS.md](RESULTS.md).

## How it works

**Crawl** (`search/crawler.py`). Seeds come from the [Tranco list](https://tranco-list.eu), a ranking of the top
million domains. Each worker process owns the hosts whose hash maps to it, so workers never need to coordinate about
URL dedup, robots.txt or politeness. Bandwidth was not the limit. The campus Wi-Fi accepted only about 120 new TCP
connections per second, and a breadth-first crawl opens a new connection for almost every page. So the scheduler
serves hosts that already have an open keep-alive connection first and lets new hosts in through a rate limiter.
That took the crawler from 46 to 321 pages/s.

**Embed** (`search/embed.py`, `search/embed_watch.py`). The model is
[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), which turns a passage into 384 numbers. It runs
through CoreML: `search/minibert.py` is a 60-line BERT that loads the Hugging Face weights exactly, and
`bench/coreml_convert.py` exports it as a fixed-shape fp16 graph. On the same GPU that is 2.2x faster than PyTorch.
bge-small reproduces its published score on the BEIR benchmark (SciFact nDCG@10 0.713). The best larger models,
bge-base and gte-modernbert, scored 2.7 and 4.3 points higher but embedded 3x and 9x slower, so bge-small stayed.

**Index** (`search/index.py`, `search/wiki.py`). The web index is a faiss HNSW graph: every vector links to its
nearest neighbours, and a search walks the graph toward the query. It is 4.4 GB and returns 99.1% of the exact top 10
in 0.13 ms. Wikipedia is too big for that: 42M vectors in fp32 would need ~65 GB of RAM. So it uses IVF-PQ. The vectors
are grouped into 16,384 clusters and each one is compressed to ~100 bytes. A search scans the 64 clusters nearest the
query, takes the top 100, and re-scores those against the full fp16 vectors, which stay on disk and are read through
mmap. That is 4.5 GB in RAM plus 33 GB on disk.

**Query** (`search/query.py`). Embed the query, search, drop results from the same page or with the same title (the
web is full of syndicated copies), attach snippets. `query serve` serves a JSON endpoint (`/search?q=...`) and a
small search page.

**Chat** (`search/chat.py`, `search/live.py`). One process holds Qwen3-0.6B (4-bit, run with MLX), the query encoder
and both indexes. For each message it:

1. fetches and indexes any URLs in the message (~30 ms to embed a page),
2. embeds the message on the Neural Engine (~1 ms),
3. searches the web and Wikipedia indexes and merges the hits by score,
4. puts up to 4 passages scoring at least 0.75 into the prompt, plus the Wikidata card of any Wikipedia article hit,
5. generates. Each chat keeps its KV cache, so a new turn only processes its own tokens.

With `--crawl` it also crawls in the background. New pages go into a small exact-search tier that is searchable
immediately. Every 50k passages that tier is sealed into an HNSW segment built in a separate process, and every 4
equal-size segments are merged into one.

Each job runs on the chip that is fastest for it and gets in the way of the others least:

| work | runs on | speed |
|---|---|---|
| crawl + HTML parsing (own process, low priority) | CPU | ~350 pages/s |
| embedding crawled passages (own process, paused while the LLM generates) | GPU | ~1,100 passages/s |
| embedding the query (32-token CoreML graph, kept awake between queries) | Neural Engine | ~1 ms |
| vector search | CPU | ~0.3-2 ms |
| LLM prefill and decode | GPU | ~100 ms to first token, ~370 tokens/s |

**Wikipedia and Wikidata** (`search/wiki.py`, `search/wikidata.py`). These come from the official dumps, not a
crawl. The Wikipedia dump is a series of separately compressed 100-page chunks, so worker processes decompress it in
parallel. Wikitext is cleaned with regexes and brace matching: templates that carry prose (dates, unit conversions,
lists) are rendered, while references, infoboxes, tables and files are dropped. Facts from infoboxes come from
Wikidata instead. The 43.5 GB Wikidata dump is filtered with ripgrep into one card per article, for example
"Nvidia (American technology company). founded: 1993; CEO: Jensen Huang; headquarters: Santa Clara; ...". Cards are
embedded into the same index as the article passages, so a question can match the card itself.

**Hourly updates** (`search/wiki_update.py`). Every hour it asks the MediaWiki API which articles were edited,
created, moved or deleted, downloads their current text and splits it into passages the same way. Then it compares
passage hashes with what is already indexed. Only new passages are embedded. Passages that disappeared are marked dead,
and search skips them. A typical hour has ~3,500 changed articles, ~1,800 new passages and ~1,700 removed ones. A
cycle usually takes about a minute, almost all of it spent fetching pages from the API (embedding takes ~2 s).

## Running it

Needs an Apple silicon Mac, [uv](https://docs.astral.sh/uv/), and about 100 GB of disk for the built indexes (web
30 GB, Wikipedia + Wikidata 68 GB), plus ~71 GB for the Wikipedia and Wikidata dumps while building.

```bash
uv sync
ulimit -n 65536

# CoreML graphs for bge-small: args are batch size and sequence length
uv run python bench/coreml_convert.py BAAI/bge-small-en-v1.5 1 32     # queries on the Neural Engine (chat)
uv run python bench/coreml_convert.py BAAI/bge-small-en-v1.5 1 64     # queries on the GPU (query serve)
uv run python bench/coreml_convert.py BAAI/bge-small-en-v1.5 16 256   # passages
uv run python bench/coreml_convert.py BAAI/bge-small-en-v1.5 32 256   # passages (embed_watch)
```

### Web index

```bash
# crawl for 10 minutes; in a second terminal, embed shards as the crawler closes them
uv run python -m search.crawler --workers 8 --n-seeds 40000 --new-host-rate 40 --time-limit 600 --out data/crawl/pages
uv run python -m search.embed_watch --backend coreml --shards "data/crawl/*.jsonl.zst" --parts data/index/parts

# merge (drops exact-duplicate passages), build the index, check recall against exact search
uv run python -m search.merge_parts --parts data/index/parts --out data/index/emb --dedupe
uv run python -m search.index build --emb data/index/emb --out data/index/idx --kinds flat hnsw --threads 12
uv run python -m search.index bench --emb data/index/emb --index data/index/idx --kinds hnsw --params 16 32 64 128

# search from the terminal, or at http://localhost:8080
uv run python -m search.query repl  --backend coreml --device gpu --emb data/index/emb --index data/index/idx.hnsw
uv run python -m search.query serve --backend coreml --device gpu --emb data/index/emb --index data/index/idx.hnsw
```

### Wikipedia + Wikidata

Download the dumps into `data/wiki/raw/`. The code expects these file names, so drop the `latest-` from the
downloaded names:

| file | from |
|---|---|
| `enwiki-pages-articles-multistream.xml.bz2` (26.9 GB) | [dumps.wikimedia.org/enwiki/latest](https://dumps.wikimedia.org/enwiki/latest/) |
| `enwiki-pages-articles-multistream-index.txt.bz2` | same |
| `enwiki-page_props.sql.gz` (0.5 GB) | same |
| `wikidata-truthy.nt.bz2` (43.5 GB) | [dumps.wikimedia.org/wikidatawiki/entities](https://dumps.wikimedia.org/wikidatawiki/entities/) (`latest-truthy.nt.bz2`) |

```bash
uv run python -m search.wiki extract --workers 12        # dump -> cleaned passages in data/wiki/shards
uv run python -m search.wikidata filter                  # Wikidata dump -> labels + 89 properties
uv run python -m search.wikidata build                   # one card per article -> data/wiki/wikidata.db + card shards
uv run python -m search.wiki embed --wait --device gpu   # a second copy with --device ane adds ~30%
uv run python -m search.wiki index                       # IVF-PQ index in data/wiki/index (incremental, rerun to add)
uv run python -m search.wiki show "Anarchism"            # print one article's passages, to check the cleaner
uv run python -m search.wikidata card "Nvidia"           # print one card

uv run python -m search.wiki_update run --loop 3600      # hourly updates; `status` shows the last cycle
# or as a launchd job (edit the paths first):
cp scripts/org.search.wikiupdate.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/org.search.wikiupdate.plist
```

### Chat

```bash
uv run python -m mlx_lm convert --hf-path Qwen/Qwen3-0.6B -q --q-bits 4 --mlx-path ~/models/Qwen3-0.6B-mlx-4bit
uv run python -m search.chat            # http://localhost:8800; add --crawl to crawl while you chat
```

`search.chat` defaults to the web index at `data/index/emb_all` / `idx_all.hnsw` (`--emb`, `--index`) and picks up the
Wikipedia index from `data/wiki/index` whenever it exists, reloading it every 60 s.

## Evaluation

- `uv run python -m search.evaluate beir`: nDCG@10 and Recall@100 on BEIR SciFact and NFCorpus, to check the embedding
  pipeline against the model's published scores.
- `uv run python -m search.evaluate self`: uses held-out crawled pages as answers, with each page's title or a random
  sentence from it as the query.
- `uv run python bench/e2e.py`: 30 realistic queries, per-stage latency, and the full result lists for reading by hand.

## Where it hits limits

- **Crawling** is capped by the network path's ~120 new connections/s, not by bandwidth or CPU. The next steps are a
  wired connection or more machines. Hosts are already split by hash, so N machines give N times the throughput.
- **Query latency** is ~85% embedding the query with a 12-layer model. Going lower needs a smaller query encoder.
- **Search quality** misses are mostly missing pages, not bad ranking. In a 10-query spot check every correct hit
  scored above 0.80 and every miss below 0.75, so the score works as a "nothing good found" signal (the chat uses
  0.75 as its cutoff).

## Layout

```
search/       crawler, embedding, index, query, chat server, Wikipedia/Wikidata import and updater
search/web/   the search page and the chat UI (plain HTML, no build step)
bench/        benchmark scripts and their raw results (bench/results/)
video/        a narrated Manim explainer of the pipeline (see video/README.md)
scripts/      launchd job for the hourly Wikipedia updater
data/         crawls, indexes and dumps (git-ignored)
```
