#!/bin/sh
# Full-episode recipe: 3 overlapping chunks (60 s overlap), then stitch. Uses venv from ../vibevoice/.venv
cd "$(dirname "$0")"; mkdir -p w
V=../vibevoice/.venv/bin/python; F=/Users/taylor/playground/transcription/episode/full_16k.wav
$V -I win.py $F 0 3000 30000 w/w50
$V -I win.py $F 2940 2520 30000 w/cB
$V -I win.py $F 5400 2000 30000 w/cC
python3 -I stitch.py
