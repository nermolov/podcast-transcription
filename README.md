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

## Code layout

`transcribe_episode.py` is a thin CLI over the `transcriber` package: `transcriber.pipeline.run(input, outdir, *, chunk_minutes, overlap_seconds,
model, context, max_tokens, words, keep_intermediate, progress)` returns the stats dict and raises `TranscribeError` on expected failures;
`progress(stage, fraction, message)` is called for `prepare`, `vibevoice` (per chunk), `parakeet`, `align`, `write`.
`transcriber/worker.py` is the per-chunk subprocess (`python -m transcriber.worker`). `transcriber/server.py` is the HTTP service (below). Tests: `python tests/test_pure.py`; server tests (fake pipeline) `nix develop -c python -m pytest tests`.

## Nix

A flake (aarch64-darwin only) provides the full toolchain from the pins in `uv.lock` (uv2nix; Python 3.13, mlx 0.32.4,
ffmpeg-headless). Nix is installed but not on PATH by default:

```sh
export PATH=/nix/var/nix/profiles/default/bin:$PATH   # fish: fish_add_path /nix/var/nix/profiles/default/bin
nix develop                         # shell with python env, ffmpeg, uv; HF_HOME defaults to ~/.cache/huggingface
nix run . -- <audio-or-url> ...     # runs transcribe_episode.py as `transcribe-episode` (ffmpeg on PATH)
nix build                           # ./result/bin/transcribe-episode
nix run .#server                    # HTTP service (see below); nix build .#server -> ./result/bin/transcribe-server
```

Flake files must be git-tracked for nix to see them. After changing deps, run `uv lock` and re-enter the shell.

## HTTP service

    STATE_DIR=~/.local/state/transcribe-service HOST=127.0.0.1 PORT=8000 API_TOKEN=secret nix run .#server

Env: `STATE_DIR` (default `~/.local/state/transcribe-service`; holds `uploads/`, `jobs/<id>/` = pipeline outdir, `jobs.db`),
`HOST` (127.0.0.1), `PORT` (8000), optional `API_TOKEN` (when set, every endpoint except `/healthz` needs `Authorization: Bearer <token>`).

Jobs run strictly one at a time on a single worker thread (MLX peaks ~31 GB per 50 min chunk); the rest wait as `queued`. State is in SQLite, so on
startup `queued` jobs are re-queued and jobs left `running` are re-queued too and resume from the chunk cache in their outdir. Uploads are
streamed to disk and deleted once the job succeeds.

| Endpoint | |
|---|---|
| `POST /jobs` | multipart with `file`, or form/JSON field `url` (mp3 or RSS; first episode). Optional `chunk_minutes`, `overlap_seconds`, `context`, `words` (true/false). 202 `{id,status}`; 400 if neither/both of file and url or bad params |
| `GET /jobs` | all jobs, newest first |
| `GET /jobs/{id}` | `status` (queued, running, done, failed), `stage` (queued, prepare, vibevoice, parakeet, align, write, done), `progress` (0..1 or null), `message`, `error`, `params`, `created_at`/`started_at`/`finished_at` |
| `GET /jobs/{id}/result?format=json\|txt\|srt\|stats` | 409 until `done` (default json) |
| `DELETE /jobs/{id}` | cancels a queued job / removes a finished or failed one and its files; 409 while running |
| `GET /healthz` | liveness and queue counts, no auth |

```sh
H="Authorization: Bearer secret"
curl -H "$H" -F file=@episode.mp3 -F context="Acme, Kubernetes" localhost:8000/jobs        # {"id":"3f2a...","status":"queued"}
curl -H "$H" -d url=https://example.com/feed.xml localhost:8000/jobs
curl -H "$H" -H 'content-type: application/json' -d '{"url":"https://x/ep.mp3","words":false}' localhost:8000/jobs
curl -H "$H" localhost:8000/jobs/3f2a...                                                    # poll
curl -H "$H" "localhost:8000/jobs/3f2a.../result?format=srt" -o episode.srt
curl -H "$H" -X DELETE localhost:8000/jobs/3f2a...
```
