#!/bin/sh
# usage: ./run.sh [wav] [outdir]
cd "$(dirname "$0")"
/usr/bin/time -l .venv/bin/python -I run.py "${1:-/Users/taylor/playground/transcription/episode/clip_10min_16k.wav}" "${2:-.}"
