"""Download the models into HF_HOME: python -m transcriber.fetch_models [--check]

Uses huggingface_hub.snapshot_download, which resumes partial downloads and skips files already cached.
Already-complete snapshots are detected offline first, so nothing touches the network when everything is cached.
--check only reports cache status (exit 1 if anything is missing)."""
import sys
from huggingface_hub import snapshot_download
from transcriber.pipeline import MODEL_DEFAULT, PARAKEET

MODELS = [MODEL_DEFAULT, PARAKEET]


def cached(repo):
    try: return snapshot_download(repo, local_files_only=True)
    except Exception: return None


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    missing = 0
    for repo in MODELS:
        path = cached(repo)
        if path: print(f"{repo}: already cached at {path}", flush=True); continue
        if check: print(f"{repo}: MISSING", flush=True); missing += 1; continue
        print(f"{repo}: downloading (resumable) ...", flush=True)
        print(f"{repo}: ready at {snapshot_download(repo)}", flush=True)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
