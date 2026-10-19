#!/bin/zsh
# Build every index variant over the embedded corpus and benchmark recall@10 vs exact + latency.
# usage: bench/index_sweep.sh data/index/emb data/index/idx
export PATH="$HOME/.local/bin:$PATH"
cd ~/Search
EMB=${1:-data/index/emb}; IDX=${2:-data/index/idx}
set -x
uv run python -m search.index build --emb $EMB --out $IDX --kinds flat hnsw hnsw_sq usearch_f16 usearch_i8 ivf --M 32 --efc 200 --threads 14
uv run python -m search.index bench --emb $EMB --index $IDX --kinds hnsw hnsw_sq usearch_f16 usearch_i8 --params 16 32 64 128 256 --k 10 --nq 1000 --threads 1 --out bench/results/index_bench_hnsw.json
uv run python -m search.index bench --emb $EMB --index $IDX --kinds ivf --params 4 8 16 32 64 --k 10 --nq 1000 --threads 1 --out bench/results/index_bench_ivf.json
