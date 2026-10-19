#!/bin/zsh
# Measure real wire throughput at the network interface while varying max_body (bytes read per page).
export PATH="$HOME/.local/bin:$PATH"
ulimit -n 65536
cd ~/Search
IF=$(route -n get default 2>/dev/null | awk '/interface/ {print $2}')
rx() { netstat -ib | awk -v i="$IF" '$1==i && $7 ~ /^[0-9]+$/ {print $7; exit}'; }
OFF=${OFF:-200000}
for cfg in "1 512 600000" "1 512 200000" "2 256 600000" "2 256 200000" "3 192 200000"; do
  read W C B <<< "$cfg"
  TAG=w${W}_c${C}_b${B}
  echo "=== $TAG ==="
  R0=$(rx); T0=$(date +%s)
  uv run python -m search.crawler --workers $W --n-seeds $((4000*W)) --seed-offset $OFF --max-pages 10000000 --concurrency $C \
     --max-body $B --time-limit 60 --out data/crawl/bw_$TAG --summary bench/results/crawl_bw_$TAG.json > bench/results/crawl_bw_$TAG.log 2>&1 &
  PID=$!
  sleep 40
  CPU=$(ps -eo %cpu,command | grep -E "search.crawler|multiprocessing" | grep -v grep | grep -v "uv run" | awk '{s+=$1} END {print s}')
  wait $PID
  R1=$(rx); T1=$(date +%s)
  OFF=$((OFF + 4000*W))
  python3 -c "
import json;d=json.load(open('bench/results/crawl_bw_$TAG.json'))
rx_mb=($R1-$R0)/1e6; dt=$T1-$T0
print(f\"pages/s {d['pages_per_s']:.1f} pages {d['pages']} decompMB {d['mbytes']:.0f} rxMB@iface {rx_mb:.0f} = {rx_mb*8/dt:.0f} Mbps over {dt}s  p50_total_ms {d['latency_ms']['total']['p50']:.0f} p50_ttfb {d['latency_ms']['ttfb']['p50']:.0f} p50_dl {d['latency_ms']['download']['p50']:.0f} hosts {d['hosts']} truncated {d['errors'].get('truncated',0)} cpu% $CPU\")"
done
echo BW_DONE
