# transcribe_episode

> [!WARNING]  
> This is vibecode. No warranty is provided.

End-to-end podcast transcription with speaker turns and word timestamps, on Apple Silicon (MLX).

    uv run transcribe_episode.py <audio file | mp3 URL | RSS URL> -o <outdir>

Needs `uv` and `ffmpeg`. `uv run` creates `.venv/` (git-ignored) from `pyproject.toml` on first use;
models (about 9 GB for VibeVoice 8-bit, 1.2 GB Parakeet, 40 MB speaker embedding) download to the Hugging Face cache.

The transcript is produced progressively, so a player can start within about 15 s instead of waiting for the whole episode
(benchmarks: `research/streaming/RESULTS.md`).

Pipeline:
1. ffmpeg -> 16 kHz mono wav (`<outdir>/work/audio_16k.wav`).
2. One worker process (`python -m transcriber.worker`, models loaded once) runs the chunk plan. New audio per chunk grows
   2 min -> 6 min -> 20 min -> `--chunk-minutes` (default 40), each chunk re-reading `--overlap-seconds` (default 30) of the
   previous one; `--fixed` uses equal `--chunk-minutes` windows instead. Per chunk: Parakeet-TDT v3 word timestamps on its new
   audio, then VibeVoice-ASR 8-bit (text, speaker turns, turn timestamps) with token streaming: each turn is emitted as soon
   as the model closes it. Finished chunks are cached in `<outdir>/chunks/`, so a rerun resumes.
3. The parent stitches turns as they arrive and appends each one to `<outdir>/partial.json` once it is final (it never changes
   afterwards; `ready_until` = audio time up to which the transcript is final):
   - Speakers: raw per-chunk labels are arbitrary. `--speakers embed` (default) matches each chunk-local label to running
     global speaker centroids by titanet-small embedding (sherpa-onnx, CPU), so it does not depend on who talks in the overlap;
     `--speakers overlap` matches labels by who both chunks say spoke in the overlap (only safe for two speakers).
   - Seams: cut at a segment start of the next chunk inside the overlap; if one segment spans the overlap, its words that repeat
     the previous chunk are dropped (no lost or doubled text).
   - Words: aligned to the Parakeet words with difflib (VibeVoice text/speakers stay authoritative; anchors within the turn
     +/- 1 s; unmatched words interpolated). `--no-words` skips Parakeet.
4. Outputs: `transcript.json` (turns with words[{w,start,end,matched}]), `transcript.txt`, `subtitles.srt`, `stats.json`
   (incl. `time_to_playback_s`: earliest playback start that never caught up with the transcript in this run), `partial.json`.

Flags: `--chunk-minutes`, `--overlap-seconds`, `--fixed`, `--speakers embed|overlap`, `--model`, `--context` (hotwords/prompt hint),
`--no-words`, `--keep-intermediate` (keeps `work/`, otherwise the wav and downloaded audio are removed; the chunk cache is always kept).

Notes: VibeVoice-ASR supports at most ~59 min per pass in mlx-audio; MLX peak is ~28 GB at 40 min, ~14 GB at 2 min.
Speed is about RTF 0.27 (about 30 min for a 108 min episode on an M4 Pro); first final turn after ~16 s.

## Code layout

`transcribe_episode.py` is a thin CLI over the `transcriber` package: `transcriber.pipeline.run(input, outdir, *, chunk_minutes, overlap_seconds,
progressive, speakers, model, context, max_tokens, words, keep_intermediate, progress)` returns the stats dict and raises `TranscribeError` on expected
failures; `progress(stage, fraction, message, **info)` is called for `prepare`, `vibevoice` (fraction = final audio / duration, info `ready_s`, `audio_s`),
`write`. `Stitcher`, `OverlapMapper`/`EmbedMapper` and `progressive_plan` in `pipeline.py` do the incremental stitching and planning.
`transcriber/worker.py` is the streaming worker. `transcriber/server.py` is the HTTP service (below). Tests: `python tests/test_pure.py`;
server tests (fake pipeline) `nix develop -c python -m pytest tests`.

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

Env: `STATE_DIR` (default `~/.local/state/transcribe-service`; holds `uploads/`, `jobs/<id>/` = pipeline outdir while running, `transcripts/<tid>/` = results,
`cache/chunks/<audio sha256>/` = VibeVoice chunk cache, `jobs.db`), `HOST` (127.0.0.1), `PORT` (8000), optional `API_TOKEN` (when set, every endpoint except
`/healthz` needs `Authorization: Bearer <token>`), `CACHE_MAX_GB` (20), optional `CACHE_TTL_DAYS`.

Jobs run strictly one at a time on a single worker thread (MLX peaks ~28 GB per 40 min chunk); the rest wait as `queued`. State is in SQLite, so on
startup `queued` jobs are re-queued and jobs left `running` are re-queued too and resume from the chunk cache. Uploads are streamed to disk (hashed
while streaming) and deleted once the job succeeds. Older databases are migrated on startup: finished jobs whose result files exist become transcripts
(sha256 unknown, URL jobs still match by URL).

Deduplication: a transcript is identified by (sha256 of the audio bytes, settings fingerprint). The fingerprint is a hash of `{model, chunk_minutes, overlap_seconds, words,
progressive, speakers, pipeline_version}`; `context` is not part of it, and a `words=true` result also satisfies a `words=false` request. `POST /jobs` checks and inserts atomically:
an existing transcript gives `200 {id, status:"done", cached:true, transcript_id}` (a new done job pointing at it; the upload is discarded), a queued/running job for
the same audio and settings gives `200 {id: <that job>, status, cached:true}`, otherwise `202 {id, status:"queued", cached:false}`. `force=true` skips all of that
(the new result replaces the old transcript). For `url` jobs the check is by `key` or normalized URL at submit, and by content hash after the download (a hit finishes the job
without running any model). The chunk cache is keyed by audio sha256 and shared across jobs, so retries and re-submits with other settings resume.

Retention: transcripts outlive jobs (`DELETE /jobs/{id}` only removes the job row). After each job, least recently used transcripts and chunk caches are evicted until the total size is
below `CACHE_MAX_GB`, and anything unused for `CACHE_TTL_DAYS` is dropped. The newest result and anything belonging to an active job are never evicted.

| Endpoint | |
|---|---|
| `POST /jobs` | multipart with `file`, or form/JSON field `url` (mp3 or RSS; first episode). Optional `chunk_minutes` (40), `overlap_seconds` (30), `progressive` (true; false = equal chunks), `speakers` (`embed` or `overlap`), `context`, `words` (true/false), `sha256` (client hash of the file; 400 if it differs from the received bytes), `key` (opaque client key, usable for lookup), `force`. 202/200 as above; 400 if neither/both of file and url or bad params |
| `GET /jobs` | all jobs, newest first |
| `GET /jobs/{id}` | `status` (queued, running, done, failed), `stage` (queued, prepare, vibevoice, write, done), `progress` (0..1 or null, within the stage; for vibevoice the final share of the audio), `overall_progress` (0..1 across stages), `queue_position` (jobs ahead, when queued), `message`, `error`, `params`, `transcript_id`, `cached`, `audio_s`, `ready_s` (audio seconds of final transcript so far), `sha256`, `key`, `created_at`/`started_at`/`finished_at` |
| `GET /jobs/{id}/partial?since=<n>` | transcript so far, for starting playback before the job is done: `{status, complete, audio_s, ready_until, speakers, n_turns, since, turns}`; `turns` are final (same format as `transcript.json`, append-only, so `since` skips the first n). Start playback once `ready_until` covers the first turn (~15 s after the job starts running on an idle M4 Pro) and keep polling; queued jobs return no turns, done jobs the whole transcript, failed 409 |
| `GET /jobs/{id}/result?format=json\|txt\|srt\|stats` | 409 until `done` (default json); 404 if the transcript was evicted |
| `DELETE /jobs/{id}` | cancels a queued or running job / removes a finished or failed one: `{id, deleted, was}`; the transcript and chunk cache stay. A running job's pipeline child is killed and the call returns once it is gone, so the next job starts with the memory free |
| `DELETE /jobs` | cancels every queued and running job in one step (for a single client taking over the server): `{cancelled: [{id, was}]}`; finished jobs stay |
| `GET /transcripts?sha256=<hex>` or `?key=<key>` | lookup with the default settings (sha256 wins): 200 `{status:"done", transcript_id, job_id, audio_s, sha256, key}` or `{status:"queued"\|"running", job_id, transcript_id:null, ...}`; 404 if unknown |
| `GET /transcripts/{tid}/result?format=...` | result files of a cached transcript |
| `DELETE /transcripts/{tid}` | drops a transcript; 409 if an active job works on the same audio |
| `GET /healthz` | liveness and queue counts, no auth |

```sh
H="Authorization: Bearer secret"
curl -H "$H" -F file=@episode.mp3 -F context="Acme, Kubernetes" localhost:8000/jobs        # {"id":"3f2a...","status":"queued","cached":false}
curl -H "$H" -d url=https://example.com/feed.xml localhost:8000/jobs
curl -H "$H" -H 'content-type: application/json' -d '{"url":"https://x/ep.mp3","words":false}' localhost:8000/jobs
curl -H "$H" localhost:8000/jobs/3f2a...                                                    # poll
curl -H "$H" "localhost:8000/jobs/3f2a.../partial?since=0"                                   # turns so far + ready_until
curl -H "$H" "localhost:8000/jobs/3f2a.../result?format=srt" -o episode.srt
curl -H "$H" "localhost:8000/transcripts?sha256=$(sha256sum episode.mp3 | cut -d' ' -f1)"      # already transcribed?
curl -H "$H" -X DELETE localhost:8000/jobs/3f2a...                                          # cancel one job
curl -H "$H" -X DELETE localhost:8000/jobs                                                  # cancel everything, then POST a fresh job
```

## Operations (launchd)

Models: `nix run .#fetch-models` downloads VibeVoice-ASR-8bit and Parakeet-TDT v3 into `HF_HOME` (resumable via `huggingface_hub`; repos
already in the cache are skipped without network access; `-- --check` only reports status). Do this once before the first job, since the
service would otherwise download ~10 GB inside the first request.

Install as a per-user LaunchAgent (builds `.#server`, keeps the store path alive with a GC root in `~/.local/share/<label>/server`, writes
`~/Library/LaunchAgents/com.local.transcribe-service.plist` with `RunAtLoad` + `KeepAlive`, bootstraps it, waits for `/healthz`):

```sh
API_TOKEN=secret PORT=8000 deploy/install.sh      # re-run to upgrade/reconfigure (idempotent)
deploy/uninstall.sh                               # stop + remove agent, plist and GC root; keeps state and logs
PURGE=1 deploy/uninstall.sh                       # ... and delete STATE_DIR and logs
tail -f ~/Library/Logs/transcribe-service/server.log
launchctl print gui/$(id -u)/com.local.transcribe-service
launchctl kickstart -k gui/$(id -u)/com.local.transcribe-service   # restart
```

Config env for install.sh (written into the plist): `STATE_DIR` (~/.local/state/transcribe-service), `HOST` (127.0.0.1), `PORT` (8000),
`API_TOKEN` (empty = no auth; the plist is mode 600), `HF_HOME` (~/.cache/huggingface), `LOG_DIR` (~/Library/Logs/transcribe-service),
`LABEL` (com.local.transcribe-service). The plist's PATH includes `/nix/var/nix/profiles/default/bin`. launchd runs the built binary directly,
so boot needs no network or flake evaluation; run `deploy/install.sh` again after changing code.

A LaunchAgent lives in the user's GUI session: it starts at login and stops at logout, so it only runs while the user is logged in
(a LaunchDaemon would be needed for pre-login, but MLX/Metal work is best done in the user session). Enable auto-login for an unattended machine.

Shutdown: on SIGTERM/SIGINT the server stops the active `transcriber.worker` child (own process group; also killed via `atexit`, and the worker
exits on its own if its parent disappears, e.g. SIGKILL) and puts the running job back to `queued`; it resumes from the chunk cache on next start.
The CLI kills its worker on SIGTERM/SIGINT/SIGHUP too. `stats.json` records the original upload filename or URL as `input`.
