"""VibeVoice-ASR (+ Parakeet words) over a chunk plan in one process, streaming results as JSON lines.

    python -m transcriber.worker WAV PLAN_JSON MODEL CONTEXT MAX_TOKENS CACHE_DIR KEY WORDS(0|1)

PLAN_JSON is [[audio_start, new_start, end], ...] in seconds. For each chunk: Parakeet on [new_start, end] (words, absolute
times), then VibeVoice on [audio_start, end] with token streaming; every segment is emitted as soon as its JSON object closes.
Each finished chunk is cached as CACHE_DIR/<KEY>_p<k>_<start>_<end>.json and replayed from there on a rerun (models load only
if some chunk is not cached). Events go to stdout (one JSON object per line; anything else the libraries print goes to stderr):
  {"ev":"chunk_start","k":k,"cached":bool}  {"ev":"words","k":k,"words":[{w,start,end}]}
  {"ev":"seg","k":k,"seg":{start,end,speaker_id,text}}   (times relative to the chunk's audio_start)
  {"ev":"chunk_end","k":k,"stats":{...}}
"""
import json, os, resource, sys, time, wave

SEG_KEYS = {"Start time": "start", "Start": "start", "End time": "end", "End": "end", "Speaker ID": "speaker_id", "Speaker": "speaker_id", "Content": "text"}
from transcriber.pipeline import PARAKEET


def chunk_path(cache_dir, key, k, s0, e0): return os.path.join(cache_dir, f"{key}_p{k}_{s0:.0f}_{e0:.0f}.json")


class SegParser:
    """Pulls complete {...} segment objects out of the model's growing JSON text."""
    def __init__(self): self.pos = 0; self.dec = json.JSONDecoder()

    def feed(self, text):
        out = []
        while True:
            i = text.find("{", self.pos)
            if i < 0: return out
            try: obj, j = self.dec.raw_decode(text, i)
            except json.JSONDecodeError: return out  # object not closed yet
            self.pos = j
            seg = {SEG_KEYS[k]: v for k, v in obj.items() if k in SEG_KEYS} if isinstance(obj, dict) else {}
            if "start" in seg and "end" in seg: out.append(seg)


def tokens_to_words(sentences, offset):
    words, cur = [], None
    for s in sentences:
        for tok in s.tokens:
            if cur is None or tok.text.startswith(" "):
                if cur: words.append(cur)
                cur = dict(w=tok.text.strip(), start=round(tok.start + offset, 3), end=round(tok.end + offset, 3))
            else:
                cur["w"] += tok.text; cur["end"] = round(tok.end + offset, 3)
    if cur: words.append(cur)
    return [w for w in words if w["w"]]


def stream(wav, plan, model_id, context, max_tokens, cache_dir, key, words, emit, m=None):
    import numpy as np
    w = wave.open(wav); sr = w.getframerate(); pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pk = None
    for k, (s0, nb, e0) in enumerate(plan):
        out = chunk_path(cache_dir, key, k, s0, e0)
        if os.path.exists(out):
            d = json.load(open(out))
            if not words or d.get("words") is not None:
                emit(ev="chunk_start", k=k, cached=True)
                if d.get("words") is not None: emit(ev="words", k=k, words=d["words"])
                for s in d["segs"]: emit(ev="seg", k=k, seg=s)
                emit(ev="chunk_end", k=k, stats=dict(d["stats"], cached=True))
                continue
        import mlx.core as mx
        if m is None:
            from mlx_audio.stt import load
            t = time.time(); m = load(model_id); load_s = time.time() - t
        else: load_s = 0.0
        emit(ev="chunk_start", k=k, cached=False)
        t0 = time.time(); wl = None
        if words:
            if pk is None:
                from parakeet_mlx import from_pretrained
                pk = from_pretrained(PARAKEET)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=cache_dir) as f:
                with wave.open(f.name, "wb") as o: o.setnchannels(1); o.setsampwidth(2); o.setframerate(sr); o.writeframes(pcm[int(nb * sr):int(e0 * sr)].tobytes())
                wl = tokens_to_words(pk.transcribe(f.name, chunk_duration=120.0, overlap_duration=15.0).sentences, nb)
            emit(ev="words", k=k, words=wl)
        t1 = time.time()
        segs, passes = [], []
        dur = (e0 - s0); off, base = 0.0, 0
        while True:  # one pass, plus up to MAX_RETRIES re-runs of the rest of the chunk after a decoding loop
            audio = pcm[int((s0 + off) * sr):int(e0 * sr)].astype(np.float32) / 32768
            st = vv_pass(m, audio, sr, context, max_tokens, lambda x: (segs.append(x), emit(ev="seg", k=k, seg=x)), off, base)
            passes.append(dict(st, offset=off))
            last = max([x["end"] for x in segs] or [off])
            if not st["looped"] or len(passes) > MAX_RETRIES or dur - last < 30: break
            off, base = last, base + LABEL_STRIDE
        t2 = time.time()
        stats = dict(start=s0, new_start=nb, dur=dur, load_s=load_s, words_s=t1 - t0, gen_s=t2 - t1, first_token_s=passes[0]["first_token_s"], rtf=(t2 - t0) / dur,
                     prompt_tokens=passes[0]["prompt_tokens"], gen_tokens=sum(x["gen_tokens"] for x in passes), hit_max_tokens=any(x["hit_max_tokens"] for x in passes),
                     looped=any(x["looped"] for x in passes), passes=passes,
                     mlx_peak_gb=mx.get_peak_memory() / 1e9, rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
        json.dump(dict(segs=segs, words=wl, stats=stats), open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        emit(ev="chunk_end", k=k, stats=dict(stats, cached=False))


LOOP_TOKENS = 2000  # no segment closed for this many tokens = decoding loop (a long turn is a few hundred tokens)
MAX_RETRIES = 2
LABEL_STRIDE = 1000  # speaker labels of retry pass n are n*1000 + label: a new pass labels speakers independently


def vv_pass(m, audio, sr, context, max_tokens, on_seg, offset=0.0, label_base=0):
    """One VibeVoice pass with token streaming; on_seg(seg) for every segment as it closes (times shifted by `offset`,
    labels by `label_base`). Stops early if no segment closes for LOOP_TOKENS tokens (greedy decoding can fall into a
    repetition loop on long audio and would otherwise run to max_tokens and never close the JSON)."""
    import mlx.core as mx
    from mlx_audio.lm.sample_utils import make_sampler
    t = time.time()
    at = m._preprocess_audio(mx.array(audio), sampling_rate=sr)
    feats = m.encode_speech(at); mx.eval(feats)
    ids, mask = m._build_prompt_tokens(feats, at.shape[1] / 24000, context or None)
    toks, sp, first, since, looped, recent, prev_end = [], SegParser(), None, 0, False, [], 0.0
    adur = len(audio) / sr
    for tok, _ in m.stream_generate(input_ids=ids, speech_features=feats, acoustic_input_mask=mask, max_tokens=max_tokens,
                                    sampler=make_sampler(0.0, 1.0, 0.0, min_tokens_to_keep=1, top_k=0)):
        tok = int(tok); toks.append(tok); since += 1
        if first is None: first = time.time() - t
        if "}" in m.tokenizer.decode([tok]):
            for x in sp.feed(m.tokenizer.decode(toks)):
                since = 0; recent = (recent + [(x.get("speaker_id"), str(x.get("text", "")).strip())])[-4:]
                st, en = float(x["start"]), float(x["end"])
                # loops that keep closing segments: the same speaker and sentence 4x in a row, time running backwards or past the audio
                if (len(recent) == 4 and len(set(recent)) == 1 and len(recent[0][1]) >= 15) or st < prev_end - 5 or st > adur + 5:
                    looped = True; break
                prev_end = max(prev_end, en)
                x = dict(x, start=round(st + offset, 2), end=round(en + offset, 2))
                if x.get("speaker_id") is not None: x["speaker_id"] = int(x["speaker_id"]) + label_base
                on_seg(x)
        if looped or since >= LOOP_TOKENS: looped = True; break
    mx.clear_cache()
    return dict(prompt_tokens=int(ids.shape[1]), gen_tokens=len(toks), first_token_s=first, gen_s=time.time() - t,
                hit_max_tokens=len(toks) >= max_tokens, looped=looped)


def die_with_parent():
    """Exit if the parent goes away (e.g. SIGKILLed), so an orphaned worker never keeps holding ~30 GB of MLX memory."""
    import threading
    ppid = os.getppid()
    def watch():
        while os.getppid() == ppid: time.sleep(2)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def main(argv):
    die_with_parent()
    out = os.fdopen(os.dup(1), "w", buffering=1); os.dup2(2, 1)  # events own the real stdout; library prints go to stderr
    def emit(**kw): out.write(json.dumps(kw) + "\n")
    wav, plan, model, context, max_tokens, cache_dir, key, words = argv
    os.makedirs(cache_dir, exist_ok=True)
    stream(wav, json.loads(plan), model, context, int(max_tokens), cache_dir, key, words == "1", emit)


if __name__ == "__main__":
    main(sys.argv[1:])
