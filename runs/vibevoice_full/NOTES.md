# VibeVoice-ASR 8-bit, full 108.5-min episode (6509 s)

## Limits (research + empirical)
- Model card: up to 60 min continuous audio / 64K tokens in one pass. mlx-audio 0.6.5 hard-trims input to 59 min (vibevoice_asr.py MAX_DURATION_SECONDS) with a warning. So 108 min cannot be done in one pass: chunking required. No native chunk-streaming for the non-streaming ASR checkpoint (the "streaming" API is only for the separate VibeVoice-ASR-Streaming 1.5B/7B checkpoints, which have no timestamps).
- Context option: `context=` / `hotwords=` string is folded into the prompt (we used "Signals and Threads, Jane Street podcast").
- Memory scales ~linearly with duration (prompt ~7.5 tokens/s of audio; ~355 output tokens/min). MLX peak: 10 min 15.2 GB, 20 min 19.2 GB, 42 min 27.6 GB, 50 min 30.8 GB. `/usr/bin/time` "peak memory footprint" is much larger (33-56 GB for the 42-50 min runs, includes Metal/compressed) so >50 min single pass would be risky on 64 GB even before the 59-min cap.
- Output limits: max_tokens=30000 set, never hit (50 min used 17.0k tokens).

## Timings (/usr/bin/time -l, other agents idle)
| run | audio | wall | gen s | RTF | gen tokens | MLX peak | RSS max | footprint peak | notes |
|---|---|---|---|---|---|---|---|---|---|
| w20 (0-20 min) | 1200 s | 310 s | 307.5 | 0.256 | 7097 | 19.2 GB | 10.7 GB | - | 80 seg, no gaps |
| w50 = chunk A (0-50 min) | 3000 s | 843 s | 839.6 | 0.280 | 17043 | 30.8 GB | 12.0 GB | 55.8 GB | 176 seg, clean |
| chunk B (49-91 min) | 2520 s | 701 s | 697.2 | 0.277 | 14686 | 27.6 GB | 11.9 GB | 54.9 GB | 136 seg |
| chunk C (90-108.5 min) | 1109 s | 275 s | 272.5 | 0.246 | 6367 | 18.6 GB | 10.7 GB | 33.6 GB | 54 seg |
| **Full episode (A+B+C)** | 6509 s | **1819 s (30.3 min)** | 1809 | **0.279** | 38096 | max 30.8 GB | max 12.0 GB | max 55.8 GB | + ~3 s model load per chunk |
(the 20-min run was a probe and not part of the final transcript.) Parakeet word-count check took 84 s.

## Recipe (run.sh -> win.py x3 -> stitch.py)
Chunks: A 0-3000 s, B 2940-5460 s, C 5400-end, i.e. 60 s overlaps. Each chunk gets timestamps shifted by its offset. Speaker mapping: for each overlap, compute time-overlap of prev (already mapped) vs current labels; if label-swap overlap > agree overlap, swap current chunk's labels. Merge: keep previous chunk segments ending before the overlap midpoint, then take current-chunk segments starting at/after the last kept end (-1 s). Uses venv at ../vibevoice/.venv.

## Speaker consistency
- Raw labels (per-chunk 0/1 assignment) are arbitrary: BOTH chunk boundaries flipped without fixing (A->B: 1.1 s agree vs 58.0 s disagree; B->C after fixing B: C raw matched A's convention, i.e. raw B->C also swapped). So 2/2 boundaries (100%) flip raw; after overlap matching 0 remain (overlaps were unambiguous, 98%+ agreement). Naive concatenation would have inverted host/guest in the middle chunk.
- Always exactly 2 speakers in every pass (ids 0 and 1) across the episode; the only non-speaker segment is "[Music]" at 0:00 (dropped in stitching). Final: 179 SPEAKER_00 (host) / 177 SPEAKER_01 (guest) segments. No third voice labelled for the intro/outro (host reads outro, correctly SPEAKER_00). Not verified with speaker embeddings (not done).

## Quality / remaining problems
- 23,115 words vs Parakeet-tdt-0.6b-v3 23,291 (-0.8%). Per-10-min ratio 0.94-1.04, no dropped sections: no gaps >3 s in the final timeline; no repetition loops (repeated lines are only short backchannels: Right/Yeah/Yep). Timestamps monotonic; segments are turn-level (up to ~40 s), so coarse.
- Seam issues: 1 duplicated short "Right." at ~2954 s (seam A/B); first-pass stitch rule lost 14 s there, fixed by the final rule. Chunk C seam clean.
- Text style differs between chunks (e.g. punctuation/commas, "uh"), normal run-to-run variance. The model's own segmentation differs between the overlapping passes, so seams are cut at segment boundaries rather than word-aligned.
- Not done: embedding-based speaker verification; WER against a reference.
