"""Build the Xanadu speaker reference from two long single VibeVoice passes, and check it.

    .venv/bin/python research/streaming/build_reference.py --pass1 output/xanadu_ref --pass2 output/xanadu_ref2 \
        --wav episode/xanadu/full_16k.wav --out output/xanadu_ref_full.json --summary research/streaming/runs_xanadu/reference.json

pass1: transcript_episode.py outdir (previous pipeline, --chunk-minutes 56 --overlap-seconds 60 on the whole episode). Its second
       chunk hit max_tokens without closing the JSON (0 segments), so only its first chunk (0-3341 s) is usable.
pass2: transcriber.worker run with plan [[3100, 3100, 6622.5]] (events.jsonl): one 58.7 min pass, linked to pass1 through the
       3100-3341 s overlap. Mapping = greedy overlap votes, confirmed by titanet centroids; a pass2 speaker matching nobody is new.
The summary (metrics only, no transcript text) records: link votes and centroid cosines, the pass1/pass2 agreement over their
overlap (the noise floor of a single long pass in this crosstalk), the decoding-loop evidence of both passes, and per-region
consistency of the reference labels with voice embeddings (which exposed the unreliable tail of pass2).
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402  (puts the repo root on sys.path)
from transcriber.pipeline import Embedder, greedy  # noqa: E402

LINK = (3100.0, 3341.2)


def spk(s): return int(str(s["speaker"])[-2:]) if isinstance(s["speaker"], str) else int(s["speaker"])


def centroids(E, segs, min_s=2.0, cap=40):
    C = {}
    for s in segs:
        if s["end"] - s["start"] >= min_s: C.setdefault(spk(s), []).append(E(s["start"], s["end"]))
    return {k: (lambda m: m / np.linalg.norm(m))(np.mean(v[:cap], 0)) for k, v in C.items()}


def consistency(E, cent, segs, lo, hi):
    """Share of speech (segments >= 3 s) whose embedding is closest to its own label's centroid."""
    ok = tot = 0.0
    for s in segs:
        if not (lo <= s["start"] < hi) or s["end"] - s["start"] < 3 or spk(s) not in cent: continue
        v = E(s["start"], s["end"]); best = max(cent, key=lambda k: float(v @ cent[k])); w = s["end"] - s["start"]
        tot += w; ok += w * (best == spk(s))
    return dict(pct=round(100 * ok / max(tot, 1), 1), seconds=round(tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", required=True); ap.add_argument("--pass2", required=True); ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--summary", required=True)
    a = ap.parse_args()
    r1 = [dict(start=s["start"], end=s["end"], speaker=spk(s), text=s["text"]) for s in json.load(open(os.path.join(a.pass1, "transcript.json")))]
    p1_chunks = [json.load(open(f))["stats"] | dict(n_segs=len(json.load(open(f))["segs"])) for f in sorted(glob.glob(os.path.join(a.pass1, "chunks", "*.json")))]
    ev = [json.loads(l) for l in open(os.path.join(a.pass2, "events.jsonl"))]
    p2_stats = next(e["stats"] for e in ev if e["ev"] == "chunk_end")
    r2 = [dict(start=e["seg"]["start"] + LINK[0], end=e["seg"]["end"] + LINK[0], speaker=int(e["seg"]["speaker_id"]), text=e["seg"]["text"])
          for e in ev if e["ev"] == "seg" and e["seg"].get("speaker_id") is not None]
    E = Embedder(a.wav, threads=4)
    # link pass2 labels to pass1 ids: overlap votes, confirmed by centroids; unmatched -> new id
    M = {}
    for x in r1:
        for y in r2:
            o = min(x["end"], y["end"], LINK[1]) - max(x["start"], y["start"], LINK[0])
            if o > 0: M[(x["speaker"], y["speaker"])] = M.get((x["speaker"], y["speaker"]), 0) + o
    c1, c2 = centroids(E, r1), centroids(E, r2)
    cos = {f"{b}->{g}": round(float(c2[b] @ c1[g]), 3) for b in c2 for g in c1}
    mp = {}
    for b in sorted(c2):  # centroid argmax, accepted when it agrees with the overlap or is unambiguous
        g = max(c1, key=lambda g: float(c2[b] @ c1[g])); s = float(c2[b] @ c1[g])
        if s >= 0.7 and g not in mp.values(): mp[b] = g
    ov = greedy(M)
    nxt = max(c1) + 1
    for b in sorted({y["speaker"] for y in r2}):
        if b not in mp: mp[b] = nxt; nxt += 1
    disagree = {b: (ov[b], mp[b]) for b in ov if ov[b] != mp[b] and M[(ov[b], b)] > 5}
    # combine: pass1 before the first pass2 segment start >= 3320 s, pass2 (relabelled) from there
    cut = min(s["start"] for s in r2 if s["start"] >= 3320)
    full = [dict(speaker=f"SPEAKER_{s['speaker']:02d}", start=s["start"], end=s["end"], text=s["text"]) for s in r1 if s["start"] < cut]
    full += [dict(speaker=f"SPEAKER_{mp[s['speaker']]:02d}", start=round(s["start"], 2), end=round(s["end"], 2), text=s["text"]) for s in r2 if s["start"] >= cut]
    for x, y in zip(full, full[1:]):
        if y["start"] < x["end"]: x["end"] = y["start"]
    json.dump(full, open(a.out, "w"), indent=1)
    # checks
    r2m = [dict(s, speaker=f"SPEAKER_{mp[s['speaker']]:02d}") for s in r2]
    floor = A.speaker_agreement(r2m, [dict(s, speaker=f"SPEAKER_{s['speaker']:02d}") for s in r1], 6622.5, 3110, 3335)
    cent = {k: v for k, v in centroids(E, [s for s in full if s["end"] < 3000], min_s=3, cap=10 ** 6).items() if k in (1, 2, 3, 4)}
    regions = [(0, 3321), (3321, 4080), (4080, 5820), (5820, 6623)]
    summary = dict(
        pass1_chunks=[{k: c[k] for k in ("start", "dur", "gen_tokens", "hit_max_tokens", "gen_s", "n_segs")} for c in p1_chunks],
        pass2=dict(audio=[LINK[0], 6622.5], n_segs=len(r2), last_seg_end=max(s["end"] for s in r2), **{k: p2_stats[k] for k in ("gen_tokens", "hit_max_tokens", "gen_s")}),
        link_overlap=LINK, link_votes_s={f"{g}<-{b}": round(v, 1) for (g, b), v in sorted(M.items(), key=lambda kv: -kv[1])},
        centroid_cos_pass2_to_pass1=cos, mapping_pass2_to_global=mp, overlap_vs_centroid_disagreements=disagree, cut_s=cut,
        speakers={f"SPEAKER_{k:02d}": round(sum(s["end"] - s["start"] for s in full if s["speaker"] == f"SPEAKER_{k:02d}")) for k in range(nxt)},
        noise_floor_pass1_vs_pass2=dict(range=[3110, 3335], speaker_agree_pct=floor[0]),
        label_embedding_consistency={f"{lo}-{hi}": consistency(E, cent, full, lo, hi) for lo, hi in regions},
        reliable_until_s=5820)
    json.dump(summary, open(a.summary, "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
