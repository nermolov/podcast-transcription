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

## Recommendation

- **Worker.** One persistent worker per job: both models loaded once, no subprocess per chunk. Memory peaks at 28.4 GB with a 40-min cap.
- **Chunk plan.** 2 min, then 6 min, then 20 min, then 40-min chunks, with a 30 s overlap (`geo3`).
- **Delivery.** Stream segments as their JSON closes. Expose a partial transcript through the API (for example, `GET /jobs/{id}/result?partial=1` returning the segments so far plus a `ready_until` timestamp). The client starts playback when `ready_until` covers the first segment, about 15 s on an idle M4 Pro.
- **Speaker identity.** Map chunk-local labels to global IDs with speaker-embedding centroids, and keep overlap matching as a cross-check (flag disagreements).
- **Stitching.** Replace `pipeline.stitch`'s cut rule with seam rule v2. The full-pass pipeline benefits too.
- **Word timestamps.** Run Parakeet after each chunk's VibeVoice pass (or lagging behind it) rather than in front of it.

## Caveats

- One two-speaker episode. Speaker mapping hit the ceiling for every method; a three-or-more-speaker episode is needed to separate them.
- The reference is a VibeVoice pass, not ground truth, so WER here measures disagreement and not accuracy.
- Best-case idle numbers. Concurrent jobs or other GPU load will cut into the headroom.
- The download step is not included (the input was a local mp3), and ffmpeg converted the whole file up front (4 s). Converting only the first chunk first would save about 3 s.
