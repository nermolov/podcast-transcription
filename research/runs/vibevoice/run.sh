#!/bin/sh
cd "$(dirname "$0")"
.venv/bin/python -I run.py /Users/taylor/playground/transcription/episode/clip_10min_16k.wav transcript
