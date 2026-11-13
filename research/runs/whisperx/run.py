#!/usr/bin/env python
# usage: .venv/bin/python -I run.py INPUT.wav OUTDIR [asr_model] [diar_device]
import sys, os, time, json, resource, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
inp, outdir = sys.argv[1], sys.argv[2]
asr_model = sys.argv[3] if len(sys.argv) > 3 else "large-v3"
ddev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
os.makedirs(outdir, exist_ok=True)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch, whisperx
from pyannote.audio import Pipeline
T = {}
t0 = time.time()
audio = whisperx.load_audio(inp)  # ffmpeg -> float32 16k mono
dur = len(audio) / 16000
# 1. ASR (faster-whisper / CTranslate2, CPU int8)
t = time.time()
model = whisperx.load_model(asr_model, "cpu", compute_type="int8", language="en")
T["asr_load"] = time.time() - t; t = time.time()
res = model.transcribe(audio, batch_size=8)
T["vad_plus_asr"] = time.time() - t; t = time.time(); print("T",T,flush=True)
# 2. forced alignment (wav2vec2)
am, meta = whisperx.load_align_model("en", "cpu")
res = whisperx.align(res["segments"], am, meta, audio, "cpu")
T["align"] = time.time() - t; t = time.time()
# 3. diarization: pyannote community-1 from ungated mirror pyannote-community/speaker-diarization-community-1
pipe = Pipeline.from_pretrained(os.path.join(HERE, "models/community-1"))
pipe.to(torch.device(ddev))
wav = {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}  # bypass broken torchcodec
out = pipe(wav, **({"num_speakers": int(os.environ["NUM_SPEAKERS"])} if os.environ.get("NUM_SPEAKERS") else {}))
ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
import pandas as pd
df = pd.DataFrame([{"start": s.start, "end": s.end, "speaker": spk} for s, _, spk in ann.itertracks(yield_label=True)])
T["diarize"] = time.time() - t; t = time.time()
# 4. assign
r = whisperx.assign_word_speakers(df, res)
T["total"] = time.time() - t0
segs = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "speaker": s.get("speaker", "UNKNOWN"), "text": s["text"].strip()} for s in r["segments"]]
def ts(x): return f"{int(x//60):02d}:{x%60:04.1f}"
# merge consecutive same-speaker segments for readability in txt
with open(os.path.join(outdir, "transcript.txt"), "w") as f:
    for s in segs: f.write(f"[{ts(s['start'])} - {ts(s['end'])}] {s['speaker']}: {s['text']}\n")
json.dump(segs, open(os.path.join(outdir, "transcript.json"), "w"), indent=1)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
info = {"audio_sec": dur, "timings_sec": T, "rtf": T["total"] / dur, "peak_rss_GB": peak, "asr_model": asr_model, "diar_device": ddev,
        "n_speakers": len(set(s["speaker"] for s in segs))}
json.dump(info, open(os.path.join(outdir, "timing.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
