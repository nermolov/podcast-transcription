import sys, json, time, os, numpy as np, soundfile as sf, sherpa_onnx as so
wav, out = sys.argv[1], sys.argv[2]
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
emb = os.environ.get("EMB", "nemo_en_titanet_small.onnx")
os.makedirs(out, exist_ok=True)
audio, sr = sf.read(wav, dtype="float32"); assert sr == 16000
dur = len(audio) / sr
t0 = time.time()
cfg = so.OfflineSpeakerDiarizationConfig(
    segmentation=so.OfflineSpeakerSegmentationModelConfig(
        pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=f"{M}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        num_threads=4),
    embedding=so.SpeakerEmbeddingExtractorConfig(model=f"{M}/{emb}", num_threads=4),
    clustering=so.FastClusteringConfig(num_clusters=int(os.environ.get("NSPK", "-1")), threshold=float(os.environ.get("THR", "0.5"))),
    min_duration_on=0.3, min_duration_off=0.5)
sd = so.OfflineSpeakerDiarization(cfg)
segs = sd.process(audio).sort_by_start_time()
t1 = time.time()
# merge adjacent same-speaker segments with small gaps
m = []
for s in segs:
    if m and m[-1][2] == s.speaker and s.start - m[-1][1] < 0.8: m[-1][1] = s.end
    else: m.append([s.start, s.end, s.speaker])
d = f"{M}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
rec = so.OfflineRecognizer.from_transducer(encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
    joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=6, model_type="nemo_transducer", decoding_method="greedy_search")
res = []
MAXC = 30.0
for st, en, sp in m:
    # split long segments into <=30s chunks
    a = st
    while a < en:
        b = min(en, a + MAXC)
        if en - b < 3: b = en
        x = audio[int(a*sr):int(b*sr)]
        s = rec.create_stream(); s.accept_waveform(sr, x); rec.decode_stream(s)
        txt = s.result.text.strip()
        if txt: res.append(dict(start=round(a,2), end=round(b,2), speaker=f"SPEAKER_{sp:02d}", text=txt))
        a = b
t2 = time.time()
# merge consecutive same-speaker output
json.dump(res, open(f"{out}/transcript.json", "w"), indent=1)
f = lambda t: f"{int(t//60):02d}:{t%60:04.1f}"
open(f"{out}/transcript.txt", "w").write("".join(f"[{f(r['start'])} - {f(r['end'])}] {r['speaker']}: {r['text']}\n" for r in res))
print(f"audio {dur:.1f}s diar {t1-t0:.1f}s asr {t2-t1:.1f}s total {t2-t0:.1f}s RTF {(t2-t0)/dur:.4f} nspk {len(set(r['speaker'] for r in res))}")
