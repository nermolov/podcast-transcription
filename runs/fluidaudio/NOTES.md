# FluidAudio (Parakeet TDT v3 + offline pyannote/VBx diarization), 10-min clip

## Versions / setup
- FluidAudio git fc74bdf (main, 2026-11-10), Swift 6.4.2, Xcode CLT at /Applications/Xcode.app. `swift build -c release --product fluidaudiocli` = 98s.
- ASR: `fluidaudiocli transcribe <wav> --model-version v3 --word-timestamps --output-json` (Parakeet TDT 0.6B v3 CoreML, FluidInference HF repo, auto-downloaded).
- Diarization: `fluidaudiocli process <wav> --mode offline --output json` (pyannote segmentation + WeSpeaker embeddings + PLDA/VBx clustering; FluidInference/speaker-diarization-coreml). Sortformer not used.
- Compute units: CLI defaults (CoreML, ANE preferred); I did not pass --encoder-compute-units and did not verify ANE residency.
- No CLI speaker-attributed transcript: `merge.py` assigns each ASR word to the max-overlap diarization segment (nearest segment if none), merges consecutive same-speaker words (gap < 2s). Labels renamed SPEAKER_00.. in order of appearance.
- Reproduce: `./run.sh clip` (fish).

## Timings (warm, models cached, /usr/bin/time -l; other agents idle)
| step | wall | user | sys | peak RSS | tool-reported |
|---|---|---|---|---|---|
| ASR | 1.83s | 3.26s | 1.32s | 150 MB | 1.67s, RTFx 360 (RTF 0.0028) |
| diarization | 1.97s | 2.18s | 0.23s | 347 MB | 1.89s, RTFx 317 (RTF 0.0032) |
| merge (python) | 0.07s | 0.05s | 0 | 16 MB | - |
Total ~3.9s for 600s audio. First (cold) run of ASR took 3:28 wall due to model download + CoreML compilation; diarization cold 10.4s.

## Gotchas
- Cold start dominated by CoreML model download/compile; later runs are fast.
- ASR JSON gives word timings (startTime/endTime); diarization JSON includes big embeddings per segment.
- Full 108-min file not run.

## Quality (informal, no reference/DER)
Output is 2 speakers (about 228s SPEAKER_00 host, 368s SPEAKER_01 guest, 26 merged segments). Transcript text is accurate and fluent, with punctuation/capitalization and keeps disfluencies ("uh", "um"); errors seen include proper nouns ("Chain Street" for Jane Street, "OCAL"/"O'Camel" for OCaml) and a stray "Aaron Powell" inserted mid-speech (likely hallucinated/mis-transcribed backchannel name). Speaker attribution is mostly right but boundaries are slightly off at turn changes: short words at the end of a turn land on the wrong speaker (e.g. "Not" at 00:57-01:02 attributed to the host, and a sentence fragment "maybe" split across turns), since ASR word times and diarization segment edges (min segment 1.0s) disagree. Backchannels ("Yeah.", "Right.") during the other speaker's turn are sometimes absorbed or mis-assigned. Very fast and light.
