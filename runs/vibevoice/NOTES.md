# VibeVoice-ASR on MLX (mlx-audio)

- Model: mlx-community/VibeVoice-ASR-8bit (Microsoft VibeVoice-ASR 9B, 8-bit, 8.9 GB on disk, snapshot 725c72e5). Alternatives: 4/5/6-bit, bf16.
- Versions: mlx-audio 0.6.5, mlx 0.32.4, Python 3.12 (uv venv ./.venv). Run: ./run.sh (python -I run.py <wav> <prefix>).
- Input: clip_10min_16k.wav (600 s). Single-pass generate(), max_tokens=16384, greedy, context="Signals and Threads, Jane Street podcast".
- Output: model emits JSON list [{Start,End,Speaker,Content}] natively (diarization+timestamps in one pass); transcript.raw.txt is the raw JSON.

## Timing (final clean run, other agents idle)
- /usr/bin/time -l: 158.4 s wall, 49.9 s user, 20.7 s sys, max RSS 10.04 GB
- script: load 1.5 s (cached), generate 156.6 s, RTF 0.261 (~3.8x realtime), MLX peak 15.2 GB, 3882 generated tokens, 45 segments.
- Full 108-min file not attempted (not requested). Single pass would likely exceed the 16k-token cap (~700 tokens/min -> ~75k tokens) and has a long-context KV cost; would need chunking (e.g. 10-15 min windows with speaker-label reconciliation).

## Gotchas
- Segment keys are capitalised (Start/End/Speaker/Content); Speaker is an int. My first run failed only because run.py used lowercase keys and imported soundfile (not installed); fixed, rerun.
- HF cache du shows tiny size because blobs are symlinked; use du -L.

## Quality
Two speakers, SPEAKER_00 (host) and SPEAKER_01 (guest), stay consistent across all 45 segments; the turn structure looks right on read-through, including short backchannels ("Which are still not dead." / "Yeah, they're still there.") split correctly. Segments are turn-level (some up to ~40 s), so timestamps are coarse (to ~0.1 s at turn boundaries only). The transcript is fluent and keeps disfluencies ("uh", "like"). The first line starts mid-sentence because the clip was cut there. No obvious hallucinations or repetition loops; I did not compare against a reference, so WER is unmeasured. Possible weak spots: proper nouns (QBasic, QNX, OCaml appear right) and sentences very occasionally trailing off ("...and then I.").
