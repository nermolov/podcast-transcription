"""Progressive (chunked, token-streamed) VibeVoice run with emission timing.

    .venv/bin/python research/streaming/bench.py --plan 180,360,720,1440,2400 --overlap 30 --out research/streaming/runs/<name>

One process: ffmpeg -> load models -> for each chunk [Parakeet words on the chunk] -> VibeVoice stream_generate. Segments are
parsed out of the token stream as soon as their JSON object closes and logged with their wall time (relative to job start),
so analyze.py can compute both chunk-granular readiness (whole chunk done) and token-streamed readiness from one run.

--plan: new-audio length per chunk in seconds; the last value repeats until the end. Chunk k decodes [b_k - overlap, b_{k+1}].
--speaker exemplar: prepend a roster of clips of the already-known global speakers (in global-id order, separated by silence)
  to every chunk after the first, and map local labels to global ids by which roster clip they were given (online, since the
  roster for chunk k+1 depends on chunk k's mapping). overlap mode does no online mapping (analyze.py does it).
"""
import argparse, json, os, re, subprocess, sys, tempfile, time, wave
import numpy as np

SR = 16000
KEYS = {"Start time": "start", "Start": "start", "End time": "end", "End": "end", "Speaker ID": "speaker_id", "Speaker": "speaker_id", "Content": "text"}


def chunk_plan(dur, lens, ov):
    bounds, b, k = [0.0], 0.0, 0
    while b < dur - 1:
        L = lens[min(k, len(lens) - 1)]
        if dur - (b + L) < max(30.0, 0.15 * L): L = dur - b  # don't leave a tiny tail chunk
        b = min(dur, b + L); bounds.append(b); k += 1
    return [(max(0.0, bounds[i] - (ov if i else 0)), bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]  # (audio start, new start, end)


class Seg:
    """Pulls complete {...} objects out of the model's growing JSON text."""
    def __init__(s): s.pos = 0; s.dec = json.JSONDecoder()

    def feed(s, text):
        out = []
        while True:
            i = text.find("{", s.pos)
            if i < 0: return out
            try: obj, j = s.dec.raw_decode(text, i)
            except json.JSONDecodeError: return out  # object not closed yet
            s.pos = j
            seg = {KEYS[k]: v for k, v in obj.items() if k in KEYS} if isinstance(obj, dict) else {}
            if "start" in seg and "end" in seg: out.append(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="episode/episode.mp3")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--overlap", type=float, default=30)
    ap.add_argument("--speaker", choices=["overlap", "exemplar"], default="overlap")
    ap.add_argument("--clip-s", type=float, default=10.0, help="exemplar clip length per known speaker")
    ap.add_argument("--words", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model", default="mlx-community/VibeVoice-ASR-8bit")
    ap.add_argument("--context", default="")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = open(os.path.join(a.out, "events.jsonl"), "w")
    T0 = time.time()
    def emit(kind, **kw):
        kw = dict(t=round(time.time() - T0, 3), kind=kind, **kw); ev.write(json.dumps(kw) + "\n"); ev.flush()
        if kind != "seg": print(json.dumps(kw), file=sys.stderr, flush=True)

    tmp = tempfile.mkdtemp(prefix="bench_"); wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.audio, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav); pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16); dur = len(pcm) / SR
    emit("prepared", audio_s=dur)

    import mlx.core as mx
    from mlx_audio.stt import load
    from mlx_audio.lm.sample_utils import make_sampler
    m = load(a.model)
    pk = None
    if a.words:
        from parakeet_mlx import from_pretrained
        pk = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    emit("loaded")

    plan = chunk_plan(dur, [float(x) for x in a.plan.split(",")], a.overlap)
    if a.max_chunks: plan = plan[:a.max_chunks]
    json.dump(dict(args=vars(a), plan=plan, audio_s=dur), open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    emit("plan", plan=plan)

    sampler = make_sampler(0.0, 1.0, 0.0, min_tokens_to_keep=1, top_k=0)
    final = []  # global-labelled segments so far (exemplar mode; absolute times)
    for k, (s0, nb, e0) in enumerate(plan):
        audio = pcm[int(s0 * SR):int(e0 * SR)].astype(np.float32) / 32768
        if pk is not None:
            t = time.time(); sf = os.path.join(tmp, "c.wav")
            with wave.open(sf, "wb") as o: o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes(pcm[int(nb * SR):int(e0 * SR)].tobytes())
            res = pk.transcribe(sf, chunk_duration=120.0, overlap_duration=15.0)
            words = [dict(w=tok.text, start=round(tok.start + nb, 3), end=round(tok.end + nb, 3)) for s in res.sentences for tok in s.tokens]
            emit("words", chunk=k, n_tokens=len(words), secs=round(time.time() - t, 2))
            json.dump(words, open(os.path.join(a.out, f"pk_tokens_{k}.json"), "w"))
        roster, pre = [], 0.0
        if a.speaker == "exemplar" and k > 0:
            clips, cur = [], 0.0
            for g in sorted({s["speaker"] for s in final}):
                c = pick_clip([s for s in final if s["speaker"] == g], final, a.clip_s)
                if c is None: continue
                cs, ce = c
                clips.append(pcm[int(cs * SR):int(ce * SR)].astype(np.float32) / 32768); clips.append(np.zeros(SR, np.float32))
                roster.append(dict(g=g, src=[cs, ce], at=[cur, cur + ce - cs])); cur += ce - cs + 1.0
            pre = cur
            audio = np.concatenate(clips + [audio]) if clips else audio
        t = time.time()
        at = m._preprocess_audio(mx.array(audio), sampling_rate=SR)
        feats = m.encode_speech(at); mx.eval(feats)
        ids, mask = m._build_prompt_tokens(feats, at.shape[1] / 24000, a.context or None)
        emit("chunk_start", chunk=k, audio=[s0, e0], new_start=nb, prefix_s=pre, roster=roster, prompt_tokens=int(ids.shape[1]), encode_s=round(time.time() - t, 2))
        toks, since, sp, raw, first, nseg = [], 0, Seg(), [], None, 0
        for tok, _ in m.stream_generate(input_ids=ids, speech_features=feats, acoustic_input_mask=mask, max_tokens=40000, sampler=sampler):
            tok = int(tok); toks.append(tok)
            if first is None: first = time.time() - t; emit("first_token", chunk=k, secs=round(first, 2))
            if "}" in m.tokenizer.decode([tok]):
                for sg in sp.feed(m.tokenizer.decode(toks)):
                    nseg += 1; raw.append(sg)
                    emit("seg", chunk=k, start=sg["start"], end=sg["end"], spk=sg.get("speaker_id"), text=sg.get("text", ""))
        text = m.tokenizer.decode(toks)
        mx.clear_cache()
        out = dict(chunk=k, audio=[s0, e0], new_start=nb, prefix_s=pre, roster=roster, segs=raw, gen_tokens=len(toks), text=text)
        json.dump(out, open(os.path.join(a.out, f"chunk_{k}.json"), "w"), indent=1)
        emit("chunk_end", chunk=k, gen_tokens=len(toks), n_segs=nseg, secs=round(time.time() - t, 2), mlx_peak_gb=round(mx.get_peak_memory() / 1e9, 2))
        if a.speaker == "exemplar":
            final = merge_exemplar(final, out, k, a.overlap)
            emit("mapped", chunk=k, mapping=out.get("mapping"), speakers=sorted({s["speaker"] for s in final}))
    json.dump(final, open(os.path.join(a.out, "exemplar_final.json"), "w"), indent=1)
    emit("done")


def pick_clip(own, allsegs, L):
    """Longest single-speaker segment (cleanest exemplar), trimmed to L s from its middle; margins avoid neighbours bleeding in."""
    best = None
    for s in own:
        st, en = s["start"] + 0.5, s["end"] - 0.5
        if en - st < 2.0: continue
        if best is None or (en - st) > (best[1] - best[0]): best = (st, en)
    if best is None: return None
    st, en = best
    if en - st > L: mid = (st + en) / 2; st, en = mid - L / 2, mid + L / 2
    return round(st, 2), round(en, 2)


def merge_exemplar(final, out, k, ov):
    """Map chunk-local labels to global ids and append the chunk's segments (absolute time) after the previous seam."""
    s0, pre = out["audio"][0], out["prefix_s"]
    segs = [dict(start=float(s["start"]), end=float(s["end"]), spk=s.get("speaker_id"), text=str(s.get("text", "")).strip()) for s in out["segs"]]
    segs = [s for s in segs if s["spk"] is not None and s["text"]]
    mp = {}
    if k > 0:
        votes = {}
        for s in segs:
            for r in out["roster"]:
                o = min(s["end"], r["at"][1]) - max(s["start"], r["at"][0])
                if o > 0: votes[(int(s["spk"]), r["g"])] = votes.get((int(s["spk"]), r["g"]), 0) + o
        used = set()
        for (l, g), o in sorted(votes.items(), key=lambda x: -x[1]):
            if l not in mp and g not in used: mp[l] = g; used.add(g)
    nxt = max([s["speaker"] for s in final], default=-1) + 1
    body = []
    for s in segs:
        if s["end"] <= pre + 0.25: continue  # roster segments
        l = int(s["spk"])
        if l not in mp: mp[l] = nxt; nxt += 1
        body.append(dict(start=round(s["start"] - pre + s0, 2), end=round(s["end"] - pre + s0, 2), speaker=mp[l], text=s["text"]))
    out["mapping"] = {str(l): g for l, g in mp.items()}
    if k == 0: return body
    nb = out["new_start"]; mid = nb - ov / 2
    keep = [s for s in final if s["end"] <= mid + 1e-6]
    cut = keep[-1]["end"] if keep else s0
    return keep + [s for s in body if s["start"] >= cut - 0.3]


if __name__ == "__main__":
    main()
