"""Every speaker-method number in RESULTS.md, regenerated from the recorded runs (metrics only, no transcript text).

    uv run --with jiwer --python .venv/bin/python python research/streaming/eval_speakers.py sat
    uv run --with jiwer --python .venv/bin/python python research/streaming/eval_speakers.py xanadu   # needs build_reference.py first

Writes research/streaming/<runs dir>/eval_summary.json. Per run: the production Stitcher replayed with the run's own timing for
overlap matching and embeddings at several vote thresholds (strict and lenient agreement, speakers found, time to playback, WER,
seam kinds); seams whose overlap holds only one speaker; same- vs cross-speaker centroid cosines. Exemplar runs are scored with
analyze.py's roster mapping and seam rule v2. For xanadu also: titanet-small vs -large per-segment separation against the
reference, and the production service run (client_probe.json + its transcript): agreement and per-chunk headroom.
Scores cover [0, reliable_until) of the reference (xanadu: 0-97 min, see reference.json).
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A, replay as R  # noqa: E402
from transcriber.pipeline import EmbedMapper, Embedder, OverlapMapper  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EPISODES = {
    "sat": dict(runs="runs", ref="output/full_episode/transcript.json", wav="episode/full_16k.wav", until=None, merge={},
                replay=["fixed5", "geo2", "geo3", "fast", "fast_nw"], exemplar=["fast_ex"]),
    "xanadu": dict(runs="runs_xanadu", ref="output/xanadu_ref_full.json", wav="episode/xanadu/full_16k.wav", until=5820.0,
                   merge={"SPEAKER_05": "SPEAKER_03"},  # lenient: the on-air clip voice counted as the producer
                   replay=["fixed5", "geo3"], exemplar=["fixed5_ex"], service="output/xanadu_service_transcript.json"),
}
THRESHOLDS = (0.45, 0.5, 0.55, 0.6)


def scores(final, ref, refm, dur, hi, ready=None):
    d = dict(speaker_agree_pct=A.speaker_agreement(final, ref, dur, 0, hi)[0], lenient_pct=A.speaker_agreement(final, refm, dur, 0, hi)[0],
             n_speakers=len({f["speaker"] for f in final}), text_vs_ref=A.wer(ref, final, 0, hi))
    if ready: d["time_to_playback_s"] = round(max(t - f["start"] for t, f in ready), 1)
    return d


def single_speaker_overlaps(chunks):
    n = 0
    for k in range(1, len(chunks)):
        a, b = chunks[k]["s0"], chunks[k - 1]["e0"]
        prev = {s["speaker"] for s in chunks[k - 1]["mapped"] if s["end"] > a + 1 and s["start"] < b}
        cur = {s["local"] for s in chunks[k]["segs"] if s["lend"] > 1 and s["lstart"] < b - a}
        n += len(prev) < 2 or len(cur) < 2
    return dict(seams=len(chunks) - 1, single_speaker_overlaps=n)


def centroid_separation(E, chunks):
    """Cosines between per-chunk label centroids: same global speaker vs different (overlap-mapped labels)."""
    cents = []
    for c in chunks:
        for l in sorted({s["local"] for s in c["mapped"]}):
            vs = [E(s["start"], s["end"]) for s in c["mapped"] if s["local"] == l and s["end"] - s["start"] >= 1.5][:20]
            if vs:
                v = np.mean(vs, 0); g = next(s["speaker"] for s in c["mapped"] if s["local"] == l); cents.append((g, v / np.linalg.norm(v)))
    same = [float(a @ b) for i, (ga, a) in enumerate(cents) for gb, b in cents[i + 1:] if ga == gb]
    diff = [float(a @ b) for i, (ga, a) in enumerate(cents) for gb, b in cents[i + 1:] if ga != gb]
    return dict(same_min=round(min(same), 3), same_mean=round(float(np.mean(same)), 3), cross_max=round(max(diff), 3), cross_mean=round(float(np.mean(diff)), 3))


def model_separation(wav, ref, hi):
    """Per-segment (>= 1.5 s) cosine to each reference speaker's centroid (first 15 segments), titanet-small vs -large."""
    from huggingface_hub import hf_hub_download
    segs = [s for s in ref if s["end"] - s["start"] >= 1.5 and s["start"] < hi]
    out = {}
    for name in ("nemo_en_titanet_small.onnx", "nemo_en_titanet_large.onnx"):
        E = Embedder(wav, model_path=hf_hub_download("csukuangfj/speaker-embedding-models", name), threads=4)
        V = [E(s["start"], s["end"]) for s in segs]; C = {}
        for v, s in zip(V, segs): C.setdefault(s["speaker"], []).append(v)
        cent = {k: (lambda m: m / np.linalg.norm(m))(np.mean(v[:15], 0)) for k, v in C.items()}
        own = np.array([float(v @ cent[s["speaker"]]) for v, s in zip(V, segs)])
        other = np.array([max(float(v @ c) for k, c in cent.items() if k != s["speaker"]) for v, s in zip(V, segs)])
        out[name] = dict(segments=len(segs), argmax_correct_pct=round(100 * float(np.mean(own > other)), 1),
                         **{f"thr_{t}": dict(own_ge=round(100 * float(np.mean(own >= t)), 1), best_other_ge=round(100 * float(np.mean(other >= t)), 1)) for t in THRESHOLDS})
    return out


def service(cfg, ref, refm, dur, hi, rdir):
    probe = json.load(open(os.path.join(rdir, "service_probe.json")))
    hyp = json.load(open(os.path.join(ROOT, cfg["service"])))
    firsts, idx = [], 0
    for t, _, n, _ in probe["timeline"]:
        while idx < n: firsts.append(t); idx += 1
    T0 = probe["time_to_playback_s"]
    plan = [(0, 120), (120, 480), (480, 1680), (1680, 4080), (4080, dur)]
    head = [round(min(T0 + hyp[i]["start"] - firsts[i] for i in range(len(hyp)) if lo <= hyp[i]["start"] < h), 1) for lo, h in plan]
    return dict(time_to_playback_s=T0, total_s=probe["total_s"], headroom_by_chunk_s=head, **scores(hyp, ref, refm, dur, hi))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode", choices=sorted(EPISODES)); a = ap.parse_args()
    cfg = EPISODES[a.episode]; rdir = os.path.join(HERE, cfg["runs"])
    ref = json.load(open(os.path.join(ROOT, cfg["ref"]))); refm = [dict(s, speaker=cfg["merge"].get(s["speaker"], s["speaker"])) for s in ref]
    wav = os.path.join(ROOT, cfg["wav"]); E = Embedder(wav, threads=4)
    out = dict(episode=a.episode, reference=cfg["ref"], scored_until_s=cfg["until"], lenient_merge=cfg["merge"], runs={})
    for name in cfg["replay"]:
        d = os.path.join(rdir, name); res = {}
        meta = json.load(open(os.path.join(d, "meta.json"))); dur = meta["audio_s"]; hi = cfg["until"] or dur
        for label, mapper in [("overlap", OverlapMapper())] + [(f"embed@{t}", EmbedMapper(E, thr=t)) for t in THRESHOLDS]:
            _, st, ready = R.replay(d, mapper)
            res[label] = dict(scores([f for _, f in ready], ref, refm, dur, hi, ready),
                              seams={s: sum(b.get("seam") == s for b in st.boundaries) for s in ("start", "start-unmatched", "dedupe")})
            print(a.episode, name, label, {k: v for k, v in res[label].items() if k != "text_vs_ref"}, flush=True)
        _, _, chunks = A.load_run(d); A.map_overlap(chunks)
        res["overlap_structure"] = single_speaker_overlaps(chunks)
        res["centroid_separation"] = centroid_separation(E, chunks)
        out["runs"][name] = res
    for name in cfg["exemplar"]:
        d = os.path.join(rdir, name); meta, _, chunks = A.load_run(d); dur = meta["audio_s"]; hi = cfg["until"] or dur
        res = {}
        for label, fn in (("overlap", A.map_overlap), ("exemplar", A.map_exemplar)):
            fn(chunks); f = A.stitch_mapped(chunks, "v2")
            res[label] = dict(scores(f, ref, refm, dur, hi), ttfp_stream_s=A.ttfp(f, chunks)[0])
            print(a.episode, name, label, {k: v for k, v in res[label].items() if k != "text_vs_ref"}, flush=True)
        out["runs"][name] = res
    if a.episode == "xanadu":
        out["embedding_model_separation"] = model_separation(wav, ref, cfg["until"])
        out["service_run"] = service(cfg, ref, refm, 6622.5, cfg["until"], rdir)
        print(json.dumps({k: out[k] for k in ("embedding_model_separation", "service_run")}, indent=1, default=float))
    json.dump(out, open(os.path.join(rdir, "eval_summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
