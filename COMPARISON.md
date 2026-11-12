# STT + diarization comparison: Signals and Threads, "The Changing Craft of Programming with John Crepezzi"

Input: 10-minute clip (starts at 2:00 of the 108-min episode), 16 kHz mono. Apple M4 Pro, 64GB.
Each system ran alone, timed with `/usr/bin/time -l`. No reference transcript, so WER/DER are not measured.

| System | Wall | RTF | Peak RSS | Speakers found | Segments |
|---|---|---|---|---|---|
| FluidAudio (Parakeet v3 + pyannote/VBx, CoreML) | 3.9 s (ASR 1.8 + diar 2.0 + merge) | ~0.006 | 347 MB | 2 (auto) | 26 |
| Parakeet v3 + Streaming Sortformer v2.1 (MLX) | 18.1 s | 0.030 | 1.31 GB | 2 (auto) | 53 |
| sherpa-onnx (Parakeet v3 int8 + pyannote-seg-3.0 + titanet) | 34.0 s | 0.057 | 3.18 GB | 16 auto; 2 when count fixed | 48 |
| VibeVoice-ASR 8-bit 9B (MLX) | 158.4 s | 0.26 | 10.0 GB (15.2 GB MLX peak) | 2 (auto) | 45 |
| WhisperX (large-v3 int8 CPU + pyannote Community-1) | 581.2 s | 0.96 | 9.61 GB | 2 (auto) | 155 |

Caveats:
- sherpa's reported result uses a speaker count I fixed at 2; auto clustering gave 16.
- WhisperX ran on CPU only (no MPS tried); diarization was 339 s of its time.
- FluidAudio compute units were CLI defaults; Neural Engine use unconfirmed.
- Full 108-min file was not run for any system.

Same stretch (~02:26) across systems: the host asks "what do you end up doing?" and the guest answers.
VibeVoice split host/guest turns most cleanly. WhisperX labelled both as SPEAKER_00. FluidAudio merged them into one segment.
See each `runs/<name>/NOTES.md` for details.
