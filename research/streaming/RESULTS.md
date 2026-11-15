# Progressive transcription: time to playback with chunked VibeVoice

Goal: a listener asks for a 1–3 h episode and can start playback in under 3 minutes, without stalling later and with speaker labels kept consistent across the whole episode.

Setup: Signals and Threads "The Changing Craft of Programming with John Crepezzi" (108.5 min, 2 speakers). Apple M4 Pro, 64 GB, machine idle, one run at a time.
Model: VibeVoice-ASR 8-bit (mlx-audio 0.6.5, mlx 0.32.4), plus Parakeet-TDT v3 for word timestamps.
Reference: `output/full_episode` (current pipeline, 3 × 37-min chunks).

How to reproduce:

- Inference: `research/streaming/run_queue.sh name:args ...` runs `bench.py`.
- Scoring: `analyze.py [--embed]` (needs jiwer, scipy and sherpa-onnx in a separate venv).

## Metric

**TTFP (time to first playback)** is the smallest delay T0 such that, if playback starts T0 after the request, every final segment is ready before the playhead reaches it. It is computed as `max_i(ready_i − start_i)`, so it covers both first-chunk latency and later stalls. Every run is scored under two readiness models:

- **chunk**: a segment becomes available when its whole chunk finishes. This is what the current subprocess-per-chunk design gives.
- **stream**: a segment becomes available when its JSON object closes in the token stream. VibeVoice emits segments in time order, so a chunk can be served while it is still decoding. The segment's chunk-to-global speaker mapping must also be known by then.

**Headroom** is the tightest margin among chunks after the first, measured when playback starts at the stream TTFP. Small headroom means a stall becomes likely under load.

**Quality vs reference:**

- Speaker agreement is time-weighted, after the best global relabelling.
- Text disagreement is word-level WER against the reference transcript. The reference is another VibeVoice pass, not ground truth.

Noise floor: two independent full passes over the same 10 minutes (`research/runs/vibevoice` vs the reference) differ by **3.3% WER** and **99.05% speaker agreement**.

## Results

`--plan` gives the new-audio length per chunk in seconds; the last value repeats. All runs use a 30 s overlap. A single process keeps both models loaded for the whole job. Parakeet runs on each chunk before VibeVoice, except in `fast_nw`.

| Run | Chunk plan (min) | Chunks | First text | **TTFP stream** | TTFP chunk | Headroom | Total | Speaker agree | WER (seam v1 → v2) |
|---|---|---|---|---|---|---|---|---|---|
| baseline (current) | 37 × 3, whole job | 3 | 1892 s | — | 1892 s (~600 s if chunk 1 were released) | — | 1892 s | ref | ref |
| fixed5 | 5, 5, 5 … | 22 | 29.0 s | 26.7 s | 90.3 s | 189 s | **1702 s** | 99.31 | 4.39 → 1.52 |
| geo2 | 2, 4, 8, 16, 32, 40 … | 7 | 18.2 s | 15.8 s | **36.2 s** | 45.5 s | 1754 s | 99.16 | 1.77 → 0.84 |
| **geo3** | 2, 6, 20, 40 … | 5 | 16.4 s | **14.1 s** | 53.5 s | 38.3 s | 1772 s | 99.24 | 1.22 → 0.84 |
| fast | 2, 15, 40 … | 5 | 16.5 s | 14.1 s | 190.6 s | **4.4 s** | 1806 s | 99.21 | 1.30 → 0.77 |
| fast_nw (no Parakeet) | 2, 15, 40 … | 5 | 16.5 s | 14.2 s | 179.4 s | 16.4 s | 1683 s | 99.21 | 1.30 → 0.77 |
| fast_ex (exemplar roster) | 2, 15, 40 … | 5 | 18.2 s | 15.9 s | 196.4 s | 0.0 s | 1789 s | 99.25 | 1.19 → 0.84 |

Word-timestamp match was 97.0–97.1% with Parakeet run per chunk, against 97.2% in the reference.

Run-to-run variance looks small: chunk 0 (2 min) took 29.87–29.97 s across all six runs. Each strategy was run only once.

## Where the time goes (2-min first chunk, measured)

| Step | Time |
|---|---|
| ffmpeg on the whole mp3 | 3.9 s |
| Model load (page cache warm) | 1.4–2.7 s |
| Parakeet on the chunk | 1.5 s |
| Audio encode (acoustic and semantic tokenizers) | 4.1 s |
| Prefill (964 tokens) | 2 s |
| First segment | +3 s |
| **First text** | **≈ 16.5 s** |

Per-second costs:

- Audio encode: ≈ 0.035 s per audio second (91 s for a 40-min chunk).
- Prefill: about 450 tok/s, at about 7.5 prompt tokens per audio second.
- Decode: about 31 tok/s, at about 6 output tokens per audio second, so the decode RTF is ≈ 0.2.
- Parakeet: ≈ 0.013 s per audio second.

Encode, prefill and Parakeet all have to finish before a chunk produces its first token. The whole-chunk RTF is ≈ 0.27, with MLX peaks of 13.6 GB at 2 min, 18.7 GB at 15 min and 28.4 GB at 40 min.

## Findings

1. **Token streaming matters most.** With segments served as they are decoded, TTFP is roughly ffmpeg + load + encode/prefill of the first chunk + one segment, about 14 s with a 2-min first chunk. Under the per-chunk model, the same plan needs 36–190 s depending on how fast the chunks grow.
2. **Growing chunks: the second chunk is the constraint.** Chunk k+1 has to produce its first new segment before the playhead reaches its new region. Encode, prefill and Parakeet scale with its length and have no output until they finish.
   - In `fast`, the 15-min second chunk arrived only 4.4 s ahead of the playhead. That's fine when idle and a stall under load.
   - About 3× growth (`geo3`) keeps more than 38 s of margin and needs only 4 seams.
   - For per-chunk delivery without token streaming, 2× growth (`geo2`) gives 36 s TTFP.
   - Chunk 3 onward can go straight to the 40-min cap.
3. **Chunking does not cost total time.** All the progressive plans finish in 1700–1810 s, against 1892 s for the full pass, because shorter chunks avoid long-context attention. Even `fixed5` is the fastest overall, but its 5-min first chunk doubles TTFP and it has 21 seams.
4. **Parakeet belongs off the critical path.** Running it before each VibeVoice chunk delays that chunk's first token: 12 s for a 15-min chunk, 30 s for a 40-min chunk. `fast_nw` regains 12 s of headroom. Deliver turns first and fill in word timestamps per chunk afterwards (about 84 s in total).
5. **Seam stitching bug (it also affects production `pipeline.stitch`).** When the next chunk opens with one long segment spanning the overlap, the cut lands about 0.35 s after its start. That segment is dropped from both chunks and 10–35 s of text disappears.
   - `fixed5` lost 868 words across 10 gaps over 4 s.
   - Seam rule v2 cuts at a segment *start* in the next chunk inside the overlap, preferring one that matches an end in the previous chunk. If there is none, it keeps the previous chunk to its end and removes the repeated leading words from the next chunk's first segment (difflib). The code is `analyze.stitch_v2`.
   - With v2, every growing plan reaches about 0.8% WER against the reference, well inside the 3.3% noise floor, with no gaps.
6. **Speaker identity across the whole episode was 99.2–99.3% for every plan and every mapping method**, at the noise floor, and no chunk dropped below 97.9%. Raw chunk-local labels flipped at 10 of 21 seams in `fixed5`, so mapping is required. On this episode the three methods are tied, but they are not equally robust:
   - **Overlap matching** (current): in 11 of 21 `fixed5` seams, the overlap contained only one speaker. The other label was matched by elimination, which works only for two speakers. With three or more speakers and short overlaps, it will mislabel.
   - **Speaker embeddings** (titanet-small via sherpa-onnx, CPU): each chunk-local label's centroid is matched to running global centroids with the Hungarian algorithm, and a new speaker is created below a cosine of 0.45.
     - Centroids of the same speaker across chunks: cosine ≥ 0.886 (mean 0.95).
     - Centroids of different speakers: cosine ≤ 0.353 (mean 0.25).
     - Cost: about 9 s of CPU per episode, which can run in parallel with the GPU.
     - It doesn't depend on who speaks in the overlap.
   - **Exemplar roster** (a 10 s clip of each known speaker, prepended in global-ID order with 1 s of silence): the model labelled the clips 0, 1, … in roster order in every chunk, so its raw labels *are* the global IDs. It costs about 22 s of extra audio per chunk (about +2 s TTFP, and headroom fell to 0 with the 15-min second chunk). Prompt length grows with the number of speakers.

## Separating the speaker methods: 3+ speakers (TrueAnon "Episode 587: Xanadu")

Setup: 110.4 min, 5 voices (2 hosts, producer, guest, a cold-open clip), plus an on-air clip voice; dense crosstalk.

Reference: no ground truth, so it is built from two independent long single passes (`replay.py --ref output/xanadu_ref_full.json`).

- **Pass 1** covers 0–55.7 min (`--chunk-minutes 56`). VibeVoice labels speakers consistently *within* a pass.
- **Pass 2** covers 51.7–110.4 min. It is linked to pass 1 through their 4-min overlap; embedding centroids confirm the link (cosine 0.87–0.97). Pass 2's extra voice (best cosine 0.56 with anyone) became speaker 5.
- **Noise floor:** where the two passes overlap, they agree on speakers for **97.0%** of speech.
- **Unreliable tail (97–110 min):** this is the end of pass 2. The reference's labels there match voice embeddings (centroids from the first half) for only 86% of speech, against 99.4% for `geo3`'s labels, so that tail is excluded from the scores.
- **Lenient score:** treats the clip voice as the producer. It may be the producer doing a character voice, and in long chunks VibeVoice itself merges the two.

Scores are speaker agreement over 0–97 min, from the production `Stitcher` replayed with each run's recorded timing (`replay.py`, `sweep.py`).

| Run | Mapping | Speaker agreement | Lenient | Speakers found | Time to playback |
|---|---|---|---|---|---|
| fixed5 (5-min chunks, 21 seams) | overlap | 72.8 | 72.8 | 10 | 30.0 s |
| fixed5_ex | exemplar roster | 92.2 | 92.2 | 8 | 28.6 s |
| fixed5 | **embed** | **92.6** | **93.9** | 8 | 30.0 s |
| geo3 (2, 6, 20, 40 min; 4 seams) | overlap | 90.0 | 91.4 | 5 | 17.8 s |
| geo3 | **embed** | **96.6** | **98.0** | 5 | 17.8 s |
| **production service run** (geo3 + embed, HTTP, idle) | embed | **97.0** | **98.4** | 5 | **20.1 s** (client view, 1 s polling) |

On Signals and Threads (two speakers), every method still scores 99.2–99.3% with the production code, so embeddings cost nothing there.

Text disagreement against the reference is 2.4–2.5% WER for every run, below the 3.3% noise floor measured on the other episode.

What separated the methods:

- **Overlap matching** falls back to elimination when the overlap holds fewer than all speakers. With three or more speakers that guess goes wrong: in `fixed5` it merged the guest into the producer and spawned 10 IDs.
- **Exemplar roster:** with 4–5 prepended clips, the model no longer reliably labels them in roster order (one chunk dropped to 89%). It also adds about 50 s of audio per chunk.
- **Embeddings** needed three refinements before they were robust here. Each came from a failure observed on this episode.
  1. **Majority votes.** Each label's segments vote for their own best match, instead of using one label centroid. VibeVoice sometimes gives two people one label inside a chunk; that mixed centroid matched nobody, and the host was declared a new speaker mid-episode (71% → 93% on `fixed5`).
  2. **Pure centroids.** Global centroids only absorb segments that match them, so mixed labels can't drift them.
  3. **Short labels.** Labels with too little speech to embed (a 0.6 s "No.") take the overlap match instead of holding back finalization (+4.4 s time to playback before this fix). Any wait is capped at 30 s of audio.
- **Thresholds:** a vote threshold of 0.45 is best (0.5–0.6 split speakers). Titanet-large separates voices no better than titanet-small (93.0% against 92.7% per-segment argmax accuracy) at 1.5× the CPU.
- **Remaining error is inside chunks, not at seams.** Both mappers score the same in the final 43-min chunk, where VibeVoice merges the clip voice into the producer's label.

### Decoding loops: a production risk found while building the reference

- **First reference pass (56-min chunks):** its second chunk generated 40,000 tokens (`max_tokens`; 38 min of GPU) without ever closing the JSON array, so the old pipeline parsed **zero** segments for that half.
- **Rerun as one 58.7-min pass:** it transcribed everything correctly up to the end of the audio, then generated junk to `max_tokens` instead of stopping. The streaming worker kept all 511 segments; the old parser would have lost them all.
- **What the worker does now:**
  - Ends a pass once a segment reaches the end of the audio.
  - Treats 2,000 tokens without a closed segment, the same speaker and sentence 4× in a row, or time running backwards as a loop.
  - Keeps what was streamed and re-decodes the rest of the chunk, up to 2 retries. Retry labels are offset by 1000 and mapped like new labels.
- **No loops were seen in chunks of 40 min or less** (all benchmark and production runs). The 40-min cap keeps well away from the regime where they appeared.

## Production validation

The nix-built HTTP service was run on an idle machine, with `client_probe.py` uploading the 110-min episode and polling `GET /jobs/{id}/partial` every second.

- **Time to playback:** the first turn appeared 20.1 s after the upload finished, and playback started then would never have stalled.
- **Headroom:** each later chunk delivered at least 63.5 s ahead of the playhead (200 s, 988 s and 2512 s for chunks 3–5).
- **Total:** 38.9 min (RTF 0.35; this episode decodes about 30% more tokens per minute than Signals and Threads).
- **Resources:** MLX peak 29.2 GB, no loops.
- **Quality:** speaker agreement 97.0% (lenient 98.4%) against a 97.0% noise floor; WER 2.4%.

## What was implemented (transcriber/)

- **Worker** (`worker.py`): one process per job, both models loaded once. Per chunk, it runs Parakeet on the new audio, then VibeVoice with token streaming. Each turn is emitted as a JSON line when its object closes. It handles loops and the end of audio as above, and caches each chunk for resume.
- **Plan** (`progressive_plan`): 2, 6, 20 min, then `chunk_minutes` (default 40) of new audio per chunk, with a 30 s overlap. A short tail is folded into the last chunk, and `--fixed` keeps equal windows.
  - Parakeet stays in front of each VibeVoice chunk, as benchmarked. With this plan its cost (5 s at chunk 2) is well inside the headroom, and words are available the moment a turn is final.
- **Stitching** (`Stitcher`): incremental and append-only, with `ready_until`. It uses seam rule v2 with word de-duplication, and maps labels through a pluggable mapper: `EmbedMapper` (default) or `OverlapMapper`.
- **Service:**
  - `GET /jobs/{id}/partial?since=n` returns final turns (append-only) plus `ready_until`.
  - Jobs report `ready_s`.
  - New params `progressive` and `speakers` are part of the fingerprint; defaults are 40-min chunks and a 30 s overlap.
- **Dependencies:** `sherpa-onnx`, `scipy` and the titanet-small model (`fetch-models`). The nix build merges `sherpa-onnx-core` into `sherpa-onnx`, the same split-wheel fix as for mlx.

## Caveats

- **Two episodes.** Speaker separation was tested on one 3+-speaker episode, and its reference is itself two VibeVoice passes. Its labels in the final 13 min were shown to be worse than the system's.
- **Text quality isn't measured.** WER here is disagreement with a VibeVoice reference, not accuracy.
- **Best-case idle numbers.** Concurrent jobs or other GPU load will cut into the headroom (at least 38 s on Signals and Threads, at least 63 s on Xanadu).
- **Measured from upload completion.** Time to playback excludes upload and download time. ffmpeg converts the whole file up front (4 s); converting only the first chunk first would save about 3 s.
