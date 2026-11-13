import sys, json, time, resource
from mlx_audio.vad import load as load_vad
from parakeet_mlx import from_pretrained
wav, out_prefix = sys.argv[1], sys.argv[2]
T0 = time.time()
asr = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
t = time.time()
res = asr.transcribe(wav, chunk_duration=120.0, overlap_duration=15.0)
t_asr = time.time() - t
words = []
for s in res.sentences:
    cur = None
    for tok in s.tokens:
        if cur is None or tok.text.startswith(" "):
            if cur: words.append(cur)
            cur = dict(w=tok.text.strip(), s=tok.start, e=tok.end)
        else:
            cur["w"] += tok.text; cur["e"] = tok.end
    if cur: words.append(cur)
words = [w for w in words if w["w"]]
t = time.time()
dia = load_vad("mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
segs = []
for o in dia.generate_stream(wav, chunk_duration=5.0, threshold=0.5):
    segs += [(g.start, g.end, g.speaker) for g in o.segments]
segs.sort()
t_dia = time.time() - t
# merge adjacent same-speaker segs
m = []
for s, e, k in segs:
    if m and m[-1][2] == k and s - m[-1][1] < 0.3: m[-1][1] = max(m[-1][1], e)
    else: m.append([s, e, k])
# assign words
def spk(w):
    best, bo = None, 0
    for s, e, k in m:
        if s > w["e"]: break  # sorted by start
        o = min(e, w["e"]) - max(s, w["s"])
        if o > bo: best, bo = k, o
    if best is None:  # nearest segment by midpoint
        mid = (w["s"] + w["e"]) / 2
        best = min(m, key=lambda x: 0 if x[0] <= mid <= x[1] else min(abs(x[0]-mid), abs(x[1]-mid)))[2]
    return best
for w in words: w["k"] = spk(w)
out = []
for w in words:
    if out and out[-1]["k"] == w["k"] and w["s"] - out[-1]["end"] < 1.0:
        out[-1]["words"].append(w["w"]); out[-1]["end"] = w["e"]
    else:
        out.append(dict(k=w["k"], start=w["s"], end=w["e"], words=[w["w"]]))
segments = [dict(start=round(o["start"],2), end=round(o["end"],2), speaker=f"SPEAKER_{o['k']:02d}", text=" ".join(o["words"])) for o in out]
ts = lambda x: f"{int(x//60):02d}:{x%60:04.1f}"
with open(out_prefix + "transcript.txt", "w") as f:
    for s in segments: f.write(f"[{ts(s['start'])} - {ts(s['end'])}] {s['speaker']}: {s['text']}\n")
json.dump(segments, open(out_prefix + "transcript.json", "w"), indent=1)
tot = time.time() - T0
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
print(f"asr {t_asr:.1f}s diar {t_dia:.1f}s total {tot:.1f}s peakRSS {mem:.2f}GB speakers {sorted(set(s['speaker'] for s in segments))} nseg {len(segments)}")
