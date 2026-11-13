#!/bin/sh
# Setup: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python whisperx huggingface_hub
#        .venv/bin/hf download pyannote-community/speaker-diarization-community-1 --local-dir models/community-1
cd "$(dirname "$0")"
.venv/bin/python -I run.py "${1:-/Users/taylor/playground/transcription/episode/clip_10min_16k.wav}" "${2:-.}" "${3:-large-v3}" "${4:-cpu}"
