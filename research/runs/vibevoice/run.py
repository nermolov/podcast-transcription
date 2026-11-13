import sys, json, time, resource
import mlx.core as mx
from mlx_audio.stt import load
audio, out = sys.argv[1], sys.argv[2]
model_id = sys.argv[3] if len(sys.argv) > 3 else "mlx-community/VibeVoice-ASR-8bit"
t0 = time.time()
model = load(model_id)
t1 = time.time()
r = model.generate(audio, max_tokens=int(sys.argv[4]) if len(sys.argv) > 4 else 16384, verbose=False,
                   context="Signals and Threads, Jane Street podcast")
t2 = time.time()
segs = getattr(r, "segments", None) or []
print("raw text head:", r.text[:300]); print("nseg", len(segs), "gen tokens", r.generation_tokens)
open(out + ".raw.txt", "w").write(r.text)
def ts(x):
    x = float(x); return f"{int(x//60):02d}:{x%60:04.1f}"
js = []
for s in segs:
    sp = s.get("Speaker", s.get("speaker_id", s.get("speaker", 0)))
    sp = f"SPEAKER_{int(sp):02d}" if str(sp).lstrip('-').isdigit() else str(sp)
    js.append(dict(start=s.get("Start", s.get("start")), end=s.get("End", s.get("end")), speaker=sp, text=(s.get("Content") or s.get("text") or "").strip()))
json.dump(js, open(out + ".json", "w"), indent=1)
with open(out + ".txt", "w") as f:
    for s in js: f.write(f"[{ts(s['start'])} - {ts(s['end'])}] {s['speaker']}: {s['text']}\n")
import wave
_w = wave.open(audio); dur = _w.getnframes()/_w.getframerate()
print(f"load {t1-t0:.1f}s gen {t2-t1:.1f}s dur {dur:.1f}s RTF {(t2-t1)/dur:.3f} peakRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9:.1f}GB mlx_peak {mx.get_peak_memory()/1e9:.1f}GB")
