# sherpa-onnx diarization + ASR (10-min clip)

Versions: sherpa-onnx 2.0.1 (pip, CPU), Python 3.12 uv venv, numpy, soundfile.
Models (from k2-fsa/sherpa-onnx GitHub releases):
- Segmentation: sherpa-onnx-pyannote-segmentation-3-0 (model.onnx fp32)
- Embedding: nemo_en_titanet_small.onnx (also downloaded, unused: 3dspeaker_eres2net_base zh-cn, silero_vad)
- ASR: sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 (greedy, transducer), 6 threads. Whisper large-v3-turbo not tried.
Pipeline: OfflineSpeakerDiarization (4 threads, min_on 0.3, min_off 0.5) -> merge same-speaker segs with gap <0.8s -> Parakeet on each segment (split into <=30s chunks).
Reproduce: `./run.sh [wav] [outdir]`, env NSPK (num clusters, default -1 = threshold) and THR (default 0.5), EMB.

## Timings (10-min clip, 600s audio; other agents were not running)
Both runs: wall 34.0s, user 152s, sys 0.75s, peak RSS ~3.18 GB; diar 21.7s, ASR 12.2s, RTF 0.057.
- Final transcript.* : NSPK=2 (cluster count given; 24+24 segments).
- transcript_auto16.*: automatic threshold 0.5 -> 16 speakers (badly over-clustered).
Full 108-min file not run (per instructions); expect ~10 min by linear extrapolation.

## Gotchas
- Release asset dir is spelled `speaker-recongition-models`. GitHub downloads were slow (~minutes).
- Auto clustering with titanet_small at threshold 0.5 over-splits (16 speakers for 2); needs num_clusters=2 or a higher THR (not tuned).
- Diarization overlap: segments are not forced non-overlapping, so adjacent timestamps overlap slightly (backchannels).
- Parakeet output has no punctuation fixes beyond its own; fillers ("uh") are kept verbatim; no word timestamps used.

## Quality assessment
With num_clusters=2, speaker turns look sensible and ASR text is accurate and fluent (punctuated, capitalised, minor errors such as the first segment "Q basic" where audio starts mid-sentence). Segments are turn-level with coarse boundaries, short interjections are sometimes merged into/overlapping a neighbour, and no accuracy (WER/DER) was measured against a reference. Automatic speaker count was unreliable, which is the main weakness; speed is excellent (~17x realtime on CPU).
