# usage: win.py <wav> <start_s> <dur_s> <max_tokens> <out_prefix>
import sys, json, time, wave, resource
import numpy as np, mlx.core as mx
from mlx_audio.stt import load
wav, st, du, mt, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
w = wave.open(wav); sr = w.getframerate(); w.setpos(int(st*sr))
a = np.frombuffer(w.readframes(int(du*sr)), dtype=np.int16).astype(np.float32)/32768
dur = len(a)/sr
t0=time.time(); m = load("mlx-community/VibeVoice-ASR-8bit"); t1=time.time()
r = m.generate(mx.array(a), sampling_rate=sr, max_tokens=mt, verbose=False, context="Signals and Threads, Jane Street podcast")
t2=time.time()
open(out+".raw.txt","w").write(r.text)
segs = r.segments or []
json.dump(segs, open(out+".segs.json","w"))
stats=dict(start=st,dur=dur,load=t1-t0,gen=t2-t1,rtf=(t2-t1)/dur,gen_tokens=r.generation_tokens,prompt_tokens=r.prompt_tokens,nseg=len(segs),
  last_end=segs[-1].get("End") if segs else None, speakers=sorted({s.get("Speaker") for s in segs}, key=str),
  rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9, mlx_peak_gb=mx.get_peak_memory()/1e9)
json.dump(stats, open(out+".stats.json","w")); print(stats)
