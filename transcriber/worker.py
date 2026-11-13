"""One VibeVoice-ASR chunk in its own process: python -m transcriber.worker WAV START DUR MODEL CONTEXT MAX_TOKENS OUT"""
import json, os, resource, sys, time, wave


def worker(wav, start, dur, model_id, context, max_tokens, out):
    import numpy as np, mlx.core as mx
    from mlx_audio.stt import load
    w = wave.open(wav); sr = w.getframerate(); w.setpos(int(start * sr))
    a = np.frombuffer(w.readframes(int(dur * sr)), dtype=np.int16).astype(np.float32) / 32768
    t0 = time.time(); m = load(model_id); t1 = time.time()
    r = m.generate(mx.array(a), sampling_rate=sr, max_tokens=max_tokens, verbose=False, context=context or None)
    t2 = time.time()
    segs = [s for s in (r.segments or []) if s.get("speaker_id") is not None and s.get("text", "").strip()]
    json.dump(dict(segs=segs, stats=dict(start=start, dur=len(a) / sr, load_s=t1 - t0, gen_s=t2 - t1, rtf=(t2 - t1) / (len(a) / sr),
        gen_tokens=r.generation_tokens, hit_max_tokens=r.generation_tokens >= max_tokens, mlx_peak_gb=mx.get_peak_memory() / 1e9,
        rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)), open(out + ".tmp", "w"))
    os.replace(out + ".tmp", out)


def die_with_parent():
    """Exit if the parent goes away (e.g. SIGKILLed), so an orphaned worker never keeps holding ~30 GB of MLX memory."""
    import threading
    ppid = os.getppid()
    def watch():
        while os.getppid() == ppid: time.sleep(2)
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


if __name__ == "__main__":
    die_with_parent()
    a = sys.argv[1:]; worker(a[0], float(a[1]), float(a[2]), a[3], a[4], int(a[5]), a[6])
