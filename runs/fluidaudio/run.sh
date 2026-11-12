#!/usr/bin/env fish
# usage: ./run.sh [clip|full]   (build once, then ASR + offline diarization + merge)
set D (dirname (status filename)); cd $D
set MODE $argv[1]; test -z "$MODE"; and set MODE clip
set IN /Users/taylor/playground/transcription/episode/(test $MODE = full; and echo full_16k.wav; or echo clip_10min_16k.wav)
test -d repo; or git clone --depth 1 https://github.com/FluidInference/FluidAudio repo
test -x repo/.build/release/fluidaudiocli; or begin; cd repo; swift build -c release --product fluidaudiocli; cd ..; end
set B $PWD/repo/.build/release/fluidaudiocli; mkdir -p out
time $B transcribe $IN --model-version v3 --word-timestamps --output-json out/asr_$MODE.json
time $B process $IN --mode offline --output out/diar_$MODE.json
python3 -I merge.py out/asr_$MODE.json out/diar_$MODE.json (test $MODE = clip; and echo transcript; or echo transcript_full)
