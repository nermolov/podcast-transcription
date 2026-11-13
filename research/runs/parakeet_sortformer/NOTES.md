# Parakeet-TDT 0.6B v3 + Streaming Sortformer v2.1 (MLX) - 10 min clip

Setup: uv venv (py3.12); parakeet-mlx 0.7.0, mlx-audio 0.6.5, mlx 0.32.4. Run: ./run.sh [wav] [out_prefix] (run.py).
Models: mlx-community/parakeet-tdt-0.6b-v3 (rev ed2b7e8), mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16 (rev e23e640; port of nvidia/diar_streaming_sortformer_4spk-v2.1).
Pipeline: Parakeet (chunk 120s/overlap 15s) token timestamps -> words; Sortformer generate_stream (5s chunks, thr 0.5) -> segments (same-speaker gaps <0.3s merged); each word -> max-overlap speaker (nearest if none); consecutive same-speaker words (gap <1s) form a line.

Timings (clip, weights pre-downloaded; download took ~4 min, untimed): wall 18.1s (RTF 0.030), user 4.8s, sys 2.8s, peak RSS 1.31GB (/usr/bin/time -l).
In-tool: ASR 8.8s (includes warmup), diarization 8.3s (incl. model load), total 17.8s. Other agents may have been running earlier; this run was alone.
Full 108-min file: not run (per orchestrator).

Gotchas: Sortformer hard cap of 4 speakers (clip used 2). Use generate_stream for long audio (offline generate attends over whole file). mlx-audio loader is mlx_audio.vad.load. Parakeet tokens are subwords; words rebuilt from leading-space tokens. No punctuation/case issues but disfluencies (uh, um) are kept; Parakeet transcribes "QBasic" as "Q Basic".

Quality: Very fast with a small memory footprint, and ASR text is clean and well punctuated. Diarization finds both speakers, but since attribution is per word, short backchannels ("kind", "origin") and overlaps get split into tiny one-word segments, giving a choppy transcript and some mid-sentence speaker flips; a min-duration smoothing pass would help. No ground truth, so DER/WER not measured.
