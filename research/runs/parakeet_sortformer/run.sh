#!/bin/sh
# usage: run.sh [wav] [out_prefix]
cd "$(dirname "$0")"
uv venv -q --python 3.12 .venv 2>/dev/null; uv pip install -q --python .venv/bin/python parakeet-mlx mlx-audio
.venv/bin/python -I run.py "${1:-/Users/taylor/playground/transcription/episode/clip_10min_16k.wav}" "${2:-}"
