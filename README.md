# Search

A search engine and question-answering chat that run on one laptop (M5 Pro MacBook, 48 GB RAM). It crawls the web,
imports English Wikipedia and Wikidata, and answers questions with a small language model (Qwen3-0.6B) that reads
the best-matching passages first. Answering a question sends nothing to an outside service.

## What it searches

| source | documents | passages |
|---|---|---|
| web crawl | 675k pages | 2.4M |
| English Wikipedia + Wikidata, updated hourly | 13.4M | 42.7M |

A passage is a ~180-word chunk of a page. Each passage is stored as an *embedding*: 384 numbers computed by the
[bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5) model, so that passages with similar meaning get similar
numbers. Searching means finding the passages whose embeddings are closest to the question's.

## Speed and accuracy

Measured on [Natural Questions](https://ai.google.com/research/NaturalQuestions) (real Google searches with known
answers), each question asked once:

| | now | before |
|---|---|---|
| finding the passages for one chat message, median / p95 | 4 / 5 ms | 23 / 61 ms |
| the right answer is in the passages given to the model | 65.9% | 60.7% |
| the model's answer is right | 38.5% | 36.6% (5% with no passages) |

p95 means 95% of questions were faster. The times are for questions asked back to back. After half a second of idle
the CPU and Neural Engine have slowed their clocks, and the same search takes ~12 ms.

"Right" means the reply contains one of the reference answers. That undercounts (a date written another way, facts
that changed since 2018), but it is fair for comparing versions. After the search, the model starts replying in
~120 ms and writes ~350 tokens/s. How each number was measured, and what else was tried, is in
[RESULTS.md](RESULTS.md).

## How a question is answered

In `search/chat.py` and `search/ann.py`:

1. **Embed the question** on the Neural Engine, Apple's on-chip accelerator for neural networks (~0.6 ms).
2. **Search all stored embeddings in one pass**: web, Wikipedia, and pages added while the app runs (45M in total).
   - The embeddings are grouped into 16,384 clusters. The question is compared with each cluster's centre, and only
     the 384 closest clusters (about 2% of the data) are searched.
   - Inside a cluster, each embedding is compressed from 768 to 96 bytes (*product quantization*: the 384 numbers are
     cut into 192 pairs, and each pair is replaced by the nearest of 16 learned values). That keeps the whole index in
     4.4 GB of RAM. The clusters are split into 8 groups that are scanned in parallel, one per CPU core.
   - The best 64 Wikipedia and 24 web candidates are scored again with their full embeddings, read from the SSD. All
     of those reads are requested at once, because one after another they take ~0.1 ms each.
3. **Build the prompt**: up to 4 passages with a similarity of at least 0.70 (1.0 = the same meaning), plus the
   Wikidata fact card of each Wikipedia article used and of names in the question.
4. **Answer** with Qwen3-0.6B (4-bit weights, run with Apple's MLX library on the GPU). Each chat keeps the model's
   *KV cache*, its saved state for the conversation so far, so a new message only processes its own words.

Checked against an exact comparison with all 42.7M Wikipedia embeddings, step 2 finds 94% of the 10 truly closest
passages (the previous index found 87%).

## How the data gets in

- **Crawl** (`search/crawler.py`). Starts from the [Tranco list](https://tranco-list.eu) of the top million
  domains. The limit was the Wi-Fi network, which allowed about 120 new connections per second, so the crawler
  prefers sites it already has a connection open to. That took it from 46 to 321 pages/s.
- **Embed** (`search/embed.py`). bge-small runs through CoreML, Apple's model runtime, compiled for a fixed input
  size: 2.2x faster than PyTorch on the same GPU, ~1,100 passages/s.
- **Wikipedia and Wikidata** (`search/wiki.py`, `search/wikidata.py`). Built from the official dumps. Wikidata becomes
  one short fact card per article ("Nvidia (American technology company). founded: 1993; CEO: Jensen Huang; ...").
- **Hourly updates** (`search/wiki_update.py`). Asks Wikipedia's API which articles changed, embeds only the passages
  that are new, and hides the ones that disappeared: about 3,500 articles and one minute per hour.
- **Live pages**. Pages from `--crawl` or from a URL pasted into the chat are searchable as soon as they are embedded.

## Running it

Needs an Apple silicon Mac, [uv](https://docs.astral.sh/uv/), and about 100 GB of disk (plus ~71 GB for the
Wikipedia and Wikidata downloads while building). The CoreML model files are created on first use.

```bash
uv sync
ulimit -n 65536

# web: crawl (here 10 minutes), embed alongside, then merge into one set of passages
uv run python -m search.crawler --workers 8 --n-seeds 40000 --new-host-rate 40 --time-limit 600 --out data/crawl/pages
uv run python -m search.embed_watch --backend coreml --shards "data/crawl/*.jsonl.zst" --parts data/index/parts
uv run python -m search.merge_parts --parts data/index/parts --out data/index/emb_all --dedupe

# Wikipedia + Wikidata (dumps in data/wiki/raw/, see below)
uv run python -m search.wiki extract
uv run python -m search.wikidata filter && uv run python -m search.wikidata build
uv run python -m search.wiki embed --wait
uv run python -m search.wiki index

# the search index and the chat
uv run python -m search.ann build                 # ~4 minutes
uv run python -m search.wikidata card-rows
uv run python -m mlx_lm convert --hf-path Qwen/Qwen3-0.6B -q --q-bits 4 --mlx-path ~/models/Qwen3-0.6B-mlx-4bit
uv run python -m search.chat                      # http://localhost:8800; --crawl also crawls while you chat

# keep Wikipedia current (scripts/org.search.wikiupdate.plist runs this as a launchd job)
uv run python -m search.wiki_update run --loop 3600
```

The dumps go in `data/wiki/raw/` without the `latest-` prefix of their download names:
`enwiki-pages-articles-multistream.xml.bz2` and its `-index.txt.bz2`, and `enwiki-page_props.sql.gz`
(from [dumps.wikimedia.org/enwiki/latest](https://dumps.wikimedia.org/enwiki/latest/)), and `wikidata-truthy.nt.bz2`
([latest-truthy.nt.bz2](https://dumps.wikimedia.org/wikidatawiki/entities/)).

## Benchmarks

| script | measures |
|---|---|
| `bench/retrieval.py` | time and answer-in-passages rate of the chat's search, per stage |
| `bench/answer_eval.py` | how often the model's answer is right, with and without passages |
| `bench/ann_truth.py`, `bench/wiki_ann.py` | how many of the truly closest passages the index finds |
| `search.evaluate beir` | the embedding model against its published scores |

## Layout

```
search/       crawler, embedding, search index, chat server, Wikipedia/Wikidata import and updater
search/web/   the chat page and a plain search page
bench/        benchmark scripts; raw results in bench/results/
video/        a narrated animated explainer of the pipeline
data/         crawls, indexes and dumps (not in git)
```
