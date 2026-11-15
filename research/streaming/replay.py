"""Replay recorded bench.py runs through the production transcriber.pipeline.Stitcher (+ mapper) with their original event timing.

    <venv with jiwer, scipy, sherpa-onnx>/bin/python research/streaming/replay.py RUN_DIR --ref REF.json --wav AUDIO_16K.wav [--mappers overlap,embed]

Scores what the shipped code would have produced and when: time to playback (max over final segments of the time the
Stitcher released it minus its start), speaker agreement and WER vs the reference, speakers found, seam kinds.
Overlap-mode runs only (exemplar runs prepend audio, which the production worker does not do).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402  (also puts the repo root on sys.path)
from transcriber.pipeline import EmbedMapper, Embedder, OverlapMapper, Stitcher  # noqa: E402


def replay(d, mapper):
    meta = json.load(open(os.path.join(d, "meta.json")))
    plan = [tuple(p) for p in meta["plan"]]
    ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    st = Stitcher(plan, mapper)
    ready = []
    segs = {k: json.load(open(os.path.join(d, f"chunk_{k}.json")))["segs"] for k in range(len(plan)) if os.path.exists(os.path.join(d, f"chunk_{k}.json"))}
    idx = {k: 0 for k in segs}
    for e in ev:
        new = []
        if e["kind"] == "seg":
            s = segs[e["chunk"]][idx[e["chunk"]]]; idx[e["chunk"]] += 1
            new = st.add(e["chunk"], s)
        elif e["kind"] == "chunk_end": new = st.chunk_done(e["chunk"])
        ready += [(e["t"], f) for f in new]
    ready += [(ev[-1]["t"], f) for f in st.finish()]
    return meta, st, ready


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--ref", default=A.REF); ap.add_argument("--wav", default=A.WAV)
    ap.add_argument("--mappers", default="overlap,embed")
    ap.add_argument("--eval", default="", help="LO,HI audio range (s) to score, e.g. where the reference is valid")
    a = ap.parse_args()
    ref = json.load(open(a.ref)); emb = None
    for d in a.runs:
        res = {}
        for name in a.mappers.split(","):
            if name == "embed": emb = emb or Embedder(a.wav, threads=4)
            meta, st, ready = replay(d, OverlapMapper() if name == "overlap" else EmbedMapper(emb))
            final = [f for _, f in ready]; dur = meta["audio_s"]
            lo, hi = map(float, a.eval.split(",")) if a.eval else (0.0, dur)
            per_chunk = [A.speaker_agreement(final, ref, dur, lo=max(p[1], lo), hi=min(p[2], hi))[0] for p in meta["plan"] if p[1] < hi and p[2] > lo]
            res[name] = dict(time_to_playback_s=round(max(t - f["start"] for t, f in ready), 1), first_final_t=ready[0][0],
                             speaker_agree_pct=A.speaker_agreement(final, ref, dur, lo, hi)[0], relabel=A.speaker_agreement(final, ref, dur, lo, hi)[1],
                             min_chunk_agree=min(per_chunk), chunks_below_95=sum(x < 95 for x in per_chunk),
                             n_speakers=len({f["speaker"] for f in final}), text_vs_ref=A.wer(ref, final, lo, hi),
                             seams={s: sum(b.get("seam") == s for b in st.boundaries) for s in ("start", "start-unmatched", "dedupe")},
                             overlap_disagreements=sum(bool(b.get("overlap_disagrees")) for b in st.boundaries),
                             low_share=sorted(v for b in st.boundaries for v in b.get("vote_share", {}).values())[:4])
            json.dump(dict(final=final, boundaries=st.boundaries), open(os.path.join(d, f"replay_{name}.json"), "w"), indent=1, default=str)
        json.dump(res, open(os.path.join(d, "replay_summary.json"), "w"), indent=1, default=float)
        print(os.path.basename(d.rstrip("/")), json.dumps(res, default=float))


if __name__ == "__main__":
    main()
