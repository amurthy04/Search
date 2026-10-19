# Results & optimization log

Machine: Apple M5 Pro (15 cores, 48 GB), macOS, campus network measured at ~280 Mbps down.
All numbers below are from `bench/results/*.json` produced by the scripts in `bench/`.

## 1. Crawler

| step | change | pages/s | notes |
|---|---|---|---|
| v1 | asyncio + aiohttp + uvloop, aiodns, selectolax, 256 conns | 46 | one core at 87% |
| profile | cProfile: 35% of CPU in `ares_getaddrinfo` blocking the loop (2.4 ms/lookup), 10% TLS handshakes, 17% extract+URL parsing | | `bench/results/prof.out` |
| v2 DNS | pure-asyncio dnspython resolver with +/- cache (`bench/dns_bench.py`: 355 lookups/s, p99 loop lag 11 ms vs 437 ms for aiodns) | 163 | TTFB p50 1.9 s -> 0.8 s because the loop stopped stalling |
| v2 MP | N worker processes, hosts sharded by xxhash; cross-worker link routing over mp.Queues | 196 (1w x 512) | 2/4/8 workers did **not** add throughput (175/166/177) and 8w x 512 collapsed the network path (0.6/s) |
| bugfix | `StreamReader.read(n)` returns whatever is buffered, not n bytes: pages were parsed partially and connections closed. Fixed with a read loop. All earlier page data discarded (`data/crawl_partialread_bug/`). | | download p50 went from 0 ms (fake) to 30 ms (real) |
| bandwidth | measured at the interface: 93-111 Mbps at 145-163 pages/s -> the link is **not** the ceiling | | `bench/crawl_bw.sh` |
| error budget | 1w x 512: 9047 pages vs 5200 failed requests; timeouts (564 x 15 s) held ~27% of connection slots | | |
| E1 timeouts | total/connect/read 15/5/8 s -> 8/3/5 s (2w x 256) | 163 -> 174 | +7%; more requests hit the deadline but slots turn over faster |
| E2 seeds | top-8000 Tranco seeds, depth 3 instead of offset-400k seeds | 135 | worse: big sites block bots (403) and time out more |
| link | `en0` is **Wi-Fi**; ping to 1.1.1.1 goes 7 ms idle -> 31 ms avg / 105 ms max under a 512-connection crawl | | the radio, not the WAN, is the shared limiter that extra workers cannot beat |
| final | 2 workers x 256 conns, 8/3/5 s timeouts, 400 KB body cap, top-150k seeds, depth 2, <=50 pages/host | ~175 | production config for the main crawl |

Where the ceiling is: Little's law. ~500 connections in flight / ~1.1 s median request (remote TTFB from thousands of
small hosts + failed requests holding slots) = ~175 pages/s. Adding in-flight connections raised latency one-for-one
(2 s -> 4 s p50 at 1.5k conns) and collapsed the path past ~4k, because the Wi-Fi link saturates on packet rate /
airtime long before its 280 Mbps bulk bandwidth. 46 -> 175 pages/s (3.8x) is the diminishing-returns point for this
machine on this link. Next levers, in order: plug into Ethernet (expected 2-3x), then more machines (the design shards
hosts by hash, so N boxes = N x throughput with no coordinator).

## 2. Embedding throughput (passages of ~180 words, max 256 tokens)

| model | dim | torch/MPS fp16 | MLX | CoreML GPU fp16 | notes |
|---|---|---|---|---|---|
| static-retrieval-mrl-en-v1 | 1024 | 22,300 (CPU) | | | no transformer; lower quality |
| all-MiniLM-L6-v2 | 384 | 1,250 | 188 | | 6 layers |
| bge-small-en-v1.5 | 384 | 690 | 159 | **1,491** (b32) / 1,201 (b64) / 637 (b128) | ANE: 266 |
| snowflake-arctic-embed-s | 384 | 713 | err | | |
| multilingual-e5-small | 384 | 520 | | | |
| bge-base-en-v1.5 | 768 | 220 | 88 | | |
| e5-base-v2 | 768 | 206 | 95 | | |
| snowflake-arctic-embed-m-v1.5 | 768 | 181 | | | |
| gte-modernbert-base | 768 | 80 | | | ModernBERT has no fast MPS attention path |
| Qwen3-Embedding-0.6B | 1024 | 25 | | | |

CoreML path: `search/minibert.py` is a 60-line traceable BERT that loads HF weights bit-exactly (cos 0.99999 vs HF);
`bench/coreml_convert.py` exports a fixed-shape fp16 mlprogram. The Neural Engine is slow for this layout (it wants
the Apple `ane_transformers` (B,C,1,S) conv layout); the GPU through CoreML is 2.2x faster than torch/MPS.

## 3. Accuracy (BEIR nDCG@10, our chunk/index pipeline)

| model | SciFact | NFCorpus |
|---|---|---|
| static-retrieval-mrl-en-v1 | 0.593 | 0.301 |
| (transformer models: pending, see below) | | |

## 4. Index
(pending)

## 5. End-to-end query latency
(pending)
