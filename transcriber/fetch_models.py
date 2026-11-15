"""Download the models into HF_HOME: python -m transcriber.fetch_models [--check]

VibeVoice-ASR and Parakeet (whole repos) plus the speaker-embedding model (one file of a multi-model repo).
Uses huggingface_hub.snapshot_download / hf_hub_download, which resume partial downloads and skip files already cached.
Already-complete snapshots are detected offline first, so nothing touches the network when everything is cached.
--check only reports cache status (exit 1 if anything is missing)."""
import sys
from huggingface_hub import hf_hub_download, snapshot_download
from transcriber.pipeline import MODEL_DEFAULT, PARAKEET, SPEAKER_MODEL

MODELS = [MODEL_DEFAULT, PARAKEET, SPEAKER_MODEL]  # repo, or (repo, filename)


def fetch(m, local_only=False):
    return hf_hub_download(*m, local_files_only=local_only) if isinstance(m, tuple) else snapshot_download(m, local_files_only=local_only)


def cached(m):
    try: return fetch(m, local_only=True)
    except Exception: return None


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    missing = 0
    for m in MODELS:
        name = "/".join(m) if isinstance(m, tuple) else m
        path = cached(m)
        if path: print(f"{name}: already cached at {path}", flush=True); continue
        if check: print(f"{name}: MISSING", flush=True); missing += 1; continue
        print(f"{name}: downloading (resumable) ...", flush=True)
        print(f"{name}: ready at {fetch(m)}", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
