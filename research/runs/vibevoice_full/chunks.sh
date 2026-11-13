#!/bin/sh
# chunk A (0-3000) done as w/w50; B and C below. 60s overlap.
cd "$(dirname "$0")"; V=/Users/taylor/playground/transcription/runs/vibevoice/.venv/bin/python
F=/Users/taylor/playground/transcription/episode/full_16k.wav
/usr/bin/time -l $V -I win.py $F 2940 2520 30000 w/cB > w/cB.log 2>&1
/usr/bin/time -l $V -I win.py $F 5400 2000 30000 w/cC > w/cC.log 2>&1
echo done > w/chunks.done
