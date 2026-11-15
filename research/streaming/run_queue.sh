#!/bin/sh
# Runs each strategy alone (sequentially) on the full episode: research/streaming/run_queue.sh name:args ...
cd "$(dirname "$0")/../.."
for spec in "$@"; do
  name=${spec%%:*}; args=${spec#*:}
  out=research/streaming/runs/$name
  [ -f "$out/events.jsonl" ] && grep -q '"done"' "$out/events.jsonl" && { echo "skip $name"; continue; }
  sleep 20
  echo "$(date +%T) start $name ($args) load: $(sysctl -n vm.loadavg)"
  .venv/bin/python research/streaming/bench.py $args --out "$out" > "research/streaming/runs/$name.log" 2>&1
  echo "$(date +%T) end $name rc=$?"
done
