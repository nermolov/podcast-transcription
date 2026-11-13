# transcribe_episode

End-to-end podcast transcription with speaker turns and word timestamps, on Apple Silicon (MLX).

    uv run transcribe_episode.py <audio file | mp3 URL | RSS URL> -o <outdir>

Needs `uv` and `ffmpeg`. `uv run` creates `.venv/` (git-ignored) from `pyproject.toml` on first use;
models (about 9 GB for VibeVoice 8-bit, 1.2 GB Parakeet) download to the Hugging Face cache.

Pipeline:
1. ffmpeg -> 16 kHz mono wav (`<outdir>/work/audio_16k.wav`).
2. VibeVoice-ASR 8-bit: text, speaker turns, coarse turn timestamps. Audio over `--chunk-minutes` (default 50) is split
   into equal windows with `--overlap-seconds` (default 60) overlap; each chunk runs in its own subprocess (memory is
   fully freed) and is cached in `<outdir>/chunks/`, so a rerun resumes. Chunks are stitched: timestamps offset, overlap
   de-duplicated at segment boundaries, speaker labels matched to the previous chunk through the overlap
   (raw per-chunk labels are arbitrary and flip).
3. Parakeet-TDT v3 over the whole file (120 s chunks, 15 s overlap) for word timestamps, then aligned to the VibeVoice
   words with difflib in ~5 min windows. VibeVoice text/speakers stay authoritative; anchors must lie within the turn
   window +/- 1 s; unmatched words are interpolated. `--no-words` skips this.
4. Outputs: `transcript.json` (turns with words[{w,start,end,matched}]), `transcript.txt`, `subtitles.srt`, `stats.json`.

Flags: `--chunk-minutes`, `--overlap-seconds`, `--model`, `--context` (hotwords/prompt hint), `--no-words`, `--keep-intermediate`
(keeps `work/`, otherwise the wav and downloaded audio are removed; the chunk cache is always kept).

Notes: VibeVoice-ASR supports at most ~59 min per pass in mlx-audio; MLX peak is ~31 GB at 50 min, ~19 GB at 20 min.
Speed is about RTF 0.28 (about 30 min for a 108 min episode on an M4 Pro).

## Nix

A flake (aarch64-darwin only) provides the full toolchain from the pins in `uv.lock` (uv2nix; Python 3.13, mlx 0.32.4,
ffmpeg-headless). Nix is installed but not on PATH by default:

```sh
export PATH=/nix/var/nix/profiles/default/bin:$PATH   # fish: fish_add_path /nix/var/nix/profiles/default/bin
nix develop                         # shell with python env, ffmpeg, uv; HF_HOME defaults to ~/.cache/huggingface
nix run . -- <audio-or-url> ...     # runs transcribe_episode.py as `transcribe-episode` (ffmpeg on PATH)
nix build                           # ./result/bin/transcribe-episode
```

Flake files must be git-tracked for nix to see them. After changing deps, run `uv lock` and re-enter the shell.
