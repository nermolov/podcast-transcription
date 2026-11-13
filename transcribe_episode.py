#!/usr/bin/env python3
"""Podcast -> speaker-attributed transcript with word timestamps (VibeVoice-ASR + Parakeet, MLX)."""
import argparse, sys
from transcriber.pipeline import MODEL_DEFAULT, TranscribeError, install_signal_handlers, run


def main():
    ap = argparse.ArgumentParser(prog="transcribe-episode", description=__doc__)
    ap.add_argument("input", help="audio file, mp3 URL or RSS URL (first episode)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--chunk-minutes", type=float, default=50.0)
    ap.add_argument("--overlap-seconds", type=float, default=60.0)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--context", default="", help="hotwords / context hint passed to VibeVoice")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--no-words", action="store_true", help="skip Parakeet word timestamps")
    ap.add_argument("--keep-intermediate", action="store_true", help="keep work/ (wav, download)")
    args = ap.parse_args()
    install_signal_handlers()
    try:
        run(args.input, args.outdir, chunk_minutes=args.chunk_minutes, overlap_seconds=args.overlap_seconds, model=args.model,
            context=args.context, max_tokens=args.max_tokens, words=not args.no_words, keep_intermediate=args.keep_intermediate)
    except TranscribeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
