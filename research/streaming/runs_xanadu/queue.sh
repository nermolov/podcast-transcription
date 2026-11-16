#!/bin/sh
# Reproduces the Xanadu (TrueAnon 587, 110 min, private feed) runs, one at a time on an idle machine (~4 h in total).
# Audio: episode/xanadu/ep.mp3 (from the feed) and episode/xanadu/full_16k.wav (ffmpeg -ac 1 -ar 16000 -c:a pcm_s16le).
# Then: build_reference.py (reference + reference.json) and eval_speakers.py xanadu (eval_summary.json).
set -e
cd "$(dirname "$0")/../../.."
R=research/streaming/runs_xanadu
# reference pass 1: the previous pipeline (before ec2b685) with 56 min chunks; only its first chunk is usable (the second loops)
[ -d output/oldref ] || git worktree add -q output/oldref d74b6c2
(cd output/oldref && ../../.venv/bin/python transcribe_episode.py ../../episode/xanadu/ep.mp3 -o ../xanadu_ref --chunk-minutes 56 --overlap-seconds 60) > $R/ref.log 2>&1
# reference pass 2: one 58.7 min streaming pass over 3100 s..end (the worker exits if its parent does, so keep a parent: sh -c)
mkdir -p output/xanadu_ref2
sh -c ".venv/bin/python -m transcriber.worker episode/xanadu/full_16k.wav '[[3100.0, 3100.0, 6622.5]]' mlx-community/VibeVoice-ASR-8bit '' 40000 output/xanadu_ref2 ref2 0 > output/xanadu_ref2/events.jsonl; true"
for spec in "fixed5:--plan 300 --overlap 30" "fixed5_ex:--plan 300 --overlap 30 --speaker exemplar" "geo3:--plan 120,360,1200,2400 --overlap 30"; do
  name=${spec%%:*}; args=${spec#*:}; sleep 20
  echo "$(date +%T) start $name"; .venv/bin/python research/streaming/bench.py --audio episode/xanadu/ep.mp3 $args --out $R/$name > $R/$name.log 2>&1; echo "$(date +%T) end $name rc=$?"
done
# production service run: STATE_DIR=output/svc_state PORT=8765 nix run .#server, then
#   .venv/bin/python research/streaming/client_probe.py episode/xanadu/ep.mp3 --url http://127.0.0.1:8765 --out $R/service_probe.json
#   curl localhost:8765/jobs/<id>/result?format=json -o output/xanadu_service_transcript.json
