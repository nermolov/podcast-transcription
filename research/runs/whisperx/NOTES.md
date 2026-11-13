# WhisperX + pyannote community-1 — 10-min clip (clip_10min_16k.wav)

## Stack / versions
whisperx 3.8.6, faster-whisper 1.2.1, ctranslate2 4.9.0, pyannote.audio 4.0.7, torch 2.8.0, Python 3.12 (uv venv).
ASR: Systran/faster-whisper-large-v3, CPU, int8, batch_size=8, language=en (forced), whisperx's built-in pyannote VAD.
Alignment: wav2vec2 English (torchaudio WAV2VEC2_ASR_BASE_960H, cpu).
Diarization: pyannote community-1 pipeline (VBx clustering + PLDA), CPU, num_speakers NOT given (auto).
Note: no HF token existed. Gated pyannote/* repos give 401, but the official ungated mirror
`pyannote-community/speaker-diarization-community-1` has full weights; used that (downloaded to models/community-1).
Everything ran on CPU (no MPS/mlx).

## Setup
uv venv --python 3.12 .venv; uv pip install whisperx huggingface_hub;
hf download pyannote-community/speaker-diarization-community-1 --local-dir models/community-1;
pre-download wav2vec2 weights (whisperx.load_align_model) and `nltk.download('punkt_tab')`. Run: ./run.sh [wav] [outdir] [model] [device]
(NUM_SPEAKERS env var optional.) Outputs: transcript.txt, transcript.json, timing.json.

## Gotchas
- torchcodec fails to load (ffmpeg dylib mismatch); run.py feeds pyannote an in-memory waveform instead.
- whisperx.align needs nltk punkt_tab; nltk 3.9.x download is blocked unless nltk.pathsec.ALLOW_PROXIED_FETCH=True.
  The first two runs crashed in align after 4 min of ASR (missing punkt_tab) — wasted time, not counted below.
- Cache of ASR is not saved, so a crash in alignment costs a re-run.

## Timings (final run, other jobs possibly concurrent earlier; orchestrator said this one ran alone)
/usr/bin/time -l: wall 581.2 s, user 1917.9 s, sys 198.1 s, max RSS 9.61 GB.
Stages: model load 2.1 s; VAD+ASR 230.1 s (not separable in whisperx); alignment 7.5 s; diarization 339.0 s (CPU, VBx);
total 578.7 s => RTF 0.96 (10 min audio in ~9.7 min). Diarization is the bottleneck; MPS for it was not tried.
Full 108-min file NOT run (RTF ~1 would be ~105 min, over the threshold).

## Quality
Auto speaker detection found 2 speakers (106 segments SPEAKER_00, 49 SPEAKER_01) — correct count. SPEAKER_01 is the host
(asks questions), SPEAKER_00 the guest (GitHub/Jane Street background). Attribution is mostly good with short backchannels
("Yeah", "Right") sensibly placed. Observed errors: at 02:26-02:28 "Got it. And then professionally, what do you end up doing?"
(host question) is labelled SPEAKER_00 (guest); a few turn boundaries are off at turn changes (e.g. 02:35 guest line
"And then I... That is like the modal experience of a startup..." merges guest start with host interjection; 00:32 similar).
Speaker labels follow segment-level majority overlap, so mixed segments take one label. ASR is generally clean; minor items:
the clip starts mid-sentence ("QBasic and just like..."), a couple of disfluency restarts are retained, and "Tumblr"/"Jane Street"/"OCaml" rendered correctly;
I did not compare against a reference, so WER is unknown. No obvious hallucinations seen on skim (only skimmed ~half the lines closely).
