#!/bin/zsh
# Sweep workers x concurrency; 75s runs; record pages/s and aggregate CPU of worker processes.
export PATH="$HOME/.local/bin:$PATH"
ulimit -n 65536
cd ~/Search
mkdir -p bench/results
OFF=100000
for cfg in "1 512" "2 512" "4 384" "8 256" "8 512" "12 384"; do
  read W C <<< "$cfg"
  echo "=== workers=$W concurrency=$C ==="
  uv run python -m search.crawler --workers $W --n-seeds $((3000*W)) --seed-offset $OFF --max-pages 10000000 --concurrency $C \
      --time-limit 75 --out data/crawl/sw_w${W}_c${C} --summary bench/results/crawl_w${W}_c${C}.json > bench/results/crawl_w${W}_c${C}.log 2>&1 &
  PID=$!
  sleep 45
  CPU=$(ps -eo %cpu,command | grep "search.crawler" | grep -v grep | grep -v "uv run" | awk '{s+=$1} END {print s}')
  echo "total_cpu%@45s=$CPU"
  wait $PID
  OFF=$((OFF + 3000*W))
  python3 -c "import json;d=json.load(open('bench/results/crawl_w${W}_c${C}.json'));print('pages/s',round(d['pages_per_s'],1),'pages',d['pages'],'MB',round(d['mbytes']),'wireMB',round(d['wire_mbytes']),'hosts',d['hosts'],'p50 total ms',round(d['latency_ms']['total']['p50']),'crashes',d['crashes'])"
  echo "$W $C $CPU" >> bench/results/crawl_sweep2_cpu.txt
done
echo SWEEP2_DONE
