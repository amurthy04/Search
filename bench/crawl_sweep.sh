#!/bin/zsh
# Sweep crawler concurrency; each run 90s; record pages/s and CPU%.
export PATH="$HOME/.local/bin:$PATH"
ulimit -n 65536
cd ~/Search
mkdir -p bench/results
for C in 256 512 1024 2048; do
  echo "=== concurrency=$C ==="
  OFF=$((RANDOM % 50000))
  ( uv run python -m search.crawler --n-seeds 10000 --seed-offset $OFF --max-pages 1000000 --concurrency $C \
      --time-limit 90 --out data/crawl/sweep_c$C --summary bench/results/crawl_c$C.json > bench/results/crawl_c$C.log 2>&1 ) &
  PID=$!
  # sample CPU of the python child
  sleep 20; CPU=$(ps -o %cpu= -p $(pgrep -P $PID -f search.crawler | head -1) 2>/dev/null); echo "cpu@20s=$CPU"
  sleep 40; CPU2=$(ps -o %cpu= -p $(pgrep -P $PID -f search.crawler | head -1) 2>/dev/null); echo "cpu@60s=$CPU2"
  wait $PID
  python3 -c "import json;d=json.load(open('bench/results/crawl_c$C.json'));print('pages/s',round(d['pages_per_s'],1),'pages',d['pages'],'MB',round(d['mbytes']),'hosts',d['hosts'])"
  echo "cpu_samples $C $CPU $CPU2" >> bench/results/crawl_sweep_cpu.txt
done
echo SWEEP_DONE
