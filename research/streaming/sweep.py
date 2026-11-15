"""Sweep EmbedMapper thresholds over recorded runs (replay.py's replay): speaker agreement, lenient agreement (ref ids in --merge
counted as one person), speakers found. python research/streaming/sweep.py --thr 0.45,0.55 RUN:REF:WAV ..."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A, replay as R  # noqa: E402
from transcriber.pipeline import EmbedMapper, Embedder, OverlapMapper  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("specs", nargs="+"); ap.add_argument("--thr", default="0.45,0.5,0.55,0.6")
ap.add_argument("--merge", default="SPEAKER_03=SPEAKER_05")
a = ap.parse_args()
mg = dict(x.split("=") for x in a.merge.split(",") if x)
embs = {}
for spec in a.specs:
    d, ref_p, wav = spec.split(":")
    ref = json.load(open(ref_p)); refm = [dict(s, speaker=mg.get(s["speaker"], s["speaker"])) for s in ref]
    emb = embs.setdefault(wav, Embedder(wav, threads=4))
    for thr in ["overlap"] + a.thr.split(","):
        meta, st, ready = R.replay(d, OverlapMapper() if thr == "overlap" else EmbedMapper(emb, thr=float(thr)))
        f = [x for _, x in ready]; dur = meta["audio_s"]
        print(f"{os.path.basename(d):8} {thr:8} agree {A.speaker_agreement(f, ref, dur)[0]:6.2f}  lenient {A.speaker_agreement(f, refm, dur)[0]:6.2f}  "
              f"speakers {len({x['speaker'] for x in f})}  ttp {max(t - x['start'] for t, x in ready):.1f}", flush=True)
