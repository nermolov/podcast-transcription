#!/usr/bin/env python3
"""Podcast -> speaker-attributed transcript with word timestamps (VibeVoice-ASR + Parakeet, MLX)."""
import argparse, difflib, hashlib, json, math, os, re, resource, shutil, subprocess, sys, time, urllib.request, wave

MODEL_DEFAULT = "mlx-community/VibeVoice-ASR-8bit"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v3"
FILL = {"uh", "um", "uhm", "mm", "hmm", "mhm", "erm", "ah"}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- worker (one VibeVoice chunk, own process)
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


# ---------------------------------------------------------------- input handling
def get_audio(src, work):
    if re.match(r"https?://", src):
        req = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}))
        r = req(src); head = r.read(512)
        if head.lstrip().startswith((b"<?xml", b"<rss", b"<feed")):
            xml = head + r.read()
            m = re.search(rb'<enclosure[^>]+url=["\']([^"\']+)', xml)
            if not m: sys.exit("RSS feed has no <enclosure> url")
            src = m.group(1).decode().replace("&amp;", "&"); log("RSS: first episode ->", src); r = req(src); head = r.read(512)
        dst = os.path.join(work, "download.audio")
        with open(dst, "wb") as f:
            f.write(head); shutil.copyfileobj(r, f)
        return dst
    if not os.path.exists(src): sys.exit(f"not found: {src}")
    return src


def to_wav(src, wav):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav); return w.getnframes() / w.getframerate()


def plan_chunks(dur, chunk_s, ov):
    if dur <= chunk_s: return [(0.0, dur)]
    n = math.ceil((dur - ov) / (chunk_s - ov))
    L = (dur + (n - 1) * ov) / n
    return [(round(k * (L - ov), 2), round(min(L, dur - k * (L - ov)), 2)) for k in range(n)]


# ---------------------------------------------------------------- VibeVoice chunks + stitching
def run_chunks(wav, plan, args, cache_dir, key):
    os.makedirs(cache_dir, exist_ok=True)
    res, infos = [], []
    for k, (st, du) in enumerate(plan):
        out = os.path.join(cache_dir, f"{key}_{k}of{len(plan)}_{st:.0f}_{du:.0f}.json")
        if os.path.exists(out):
            log(f"chunk {k+1}/{len(plan)} [{st:.0f}-{st+du:.0f}s]: cached"); d = json.load(open(out)); d["stats"]["cached"] = True
        else:
            log(f"chunk {k+1}/{len(plan)} [{st:.0f}-{st+du:.0f}s]: running VibeVoice ...")
            t = time.time()
            subprocess.run([sys.executable, os.path.abspath(__file__), "--_worker", wav, str(st), str(du), args.model, args.context or "",
                            str(args.max_tokens), out], check=True)
            d = json.load(open(out)); d["stats"]["cached"] = False
            log(f"chunk {k+1}/{len(plan)} done in {time.time()-t:.0f}s, RTF {d['stats']['rtf']:.3f}, {d['stats']['gen_tokens']} tokens, MLX peak {d['stats']['mlx_peak_gb']:.1f} GB")
            if d["stats"]["hit_max_tokens"]: log("WARNING: chunk hit max_tokens, output truncated; raise --max-tokens")
        res.append(d["segs"]); infos.append(d["stats"])
    return res, infos


def stitch(chunk_segs, plan, ov):
    allc = []
    for segs, (off, _) in zip(chunk_segs, plan):
        allc.append([dict(start=round(s["start"] + off, 2), end=round(s["end"] + off, 2), speaker=int(s["speaker_id"]), text=s["text"].strip()) for s in segs])
    flips = []; nxt = 100
    for k in range(1, len(allc)):
        a = plan[k][0]; b = plan[k - 1][0] + plan[k - 1][1]
        M = {}
        for s1 in allc[k - 1]:
            for s2 in allc[k]:
                o = min(s1["end"], s2["end"], b) - max(s1["start"], s2["start"], a)
                if o > 0: M[(s1["speaker"], s2["speaker"])] = M.get((s1["speaker"], s2["speaker"]), 0) + o
        # greedy assignment: current label -> previous label with the largest overlap
        mp, used = {}, set()
        for (p, c), o in sorted(M.items(), key=lambda x: -x[1]):
            if c not in mp and p not in used: mp[c] = p; used.add(p)
        prev_labels = {s["speaker"] for s in allc[k - 1]}
        for c in sorted({s["speaker"] for s in allc[k]}):
            if c not in mp:
                free = sorted(prev_labels - used)
                if free: mp[c] = free[0]; used.add(free[0])
                else: mp[c] = nxt; nxt += 1
        agree = sum(o for (p, c), o in M.items() if mp.get(c) == p); tot = sum(M.values())
        flipped = any(c != p for c, p in mp.items())
        flips.append(dict(boundary=k, overlap=[a, round(b, 1)], overlap_agree_s=round(agree, 1), overlap_total_s=round(tot, 1), raw_labels_flipped=flipped, mapping=mp))
        for s in allc[k]: s["speaker"] = mp[s["speaker"]]
    # seam cuts: for each boundary pick a time T where chunk k has a segment end and chunk k+1 has a segment edge
    # (so the cut is clean in both), closest to the overlap midpoint; keep chunk k up to T, chunk k+1 from T.
    cuts = []
    for k in range(len(allc) - 1):
        a = plan[k + 1][0]; b = plan[k][0] + plan[k][1]; mid = (a + b) / 2
        edges = [x for s in allc[k + 1] for x in (s["start"], s["end"])]
        cands = [s["end"] for s in allc[k] if a + 1 <= s["end"] <= b and any(abs(s["end"] - e) <= 1.0 for e in edges)]
        if not cands: cands = [s["end"] for s in allc[k] if a <= s["end"] <= b] or [mid]
        cuts.append(min(cands, key=lambda x: abs(x - mid)))
    final = []
    for k, c in enumerate(allc):
        lo = cuts[k - 1] - 0.3 if k > 0 else -1; hi = cuts[k] + 1e-6 if k < len(cuts) else 1e12
        final += [s for s in c if s["start"] >= lo and s["end"] <= hi]
    final.sort(key=lambda s: s["start"])
    for a, b in zip(final, final[1:]):
        if b["start"] < a["end"]: a["end"] = b["start"]
    return final, flips


# ---------------------------------------------------------------- Parakeet words + alignment
def parakeet_words(wav):
    from parakeet_mlx import from_pretrained
    res = from_pretrained(PARAKEET).transcribe(wav, chunk_duration=120.0, overlap_duration=15.0)
    words = []
    for s in res.sentences:
        cur = None
        for tok in s.tokens:
            if cur is None or tok.text.startswith(" "):
                if cur: words.append(cur)
                cur = dict(w=tok.text.strip(), start=round(tok.start, 3), end=round(tok.end, 3))
            else:
                cur["w"] += tok.text; cur["end"] = round(tok.end, 3)
        if cur: words.append(cur)
    return [w for w in words if w["w"]]


def norm(w): return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def align(vv, pw, margin=1.0, window_s=300.0):
    """Windowed difflib alignment of VibeVoice words to Parakeet words; returns turns with per-word timestamps."""
    pn = [norm(w["w"]) for w in pw]
    pstart = [w["start"] for w in pw]
    gv = [(ti, wi, norm(x)) for ti, t in enumerate(vv) for wi, x in enumerate(t["text"].split())]
    GA = {}
    # windows of consecutive turns (~window_s of audio each)
    wins, cur, ws0 = [], [], None
    for ti, t in enumerate(vv):
        if ws0 is None: ws0 = t["start"]
        cur.append(ti)
        if t["end"] - ws0 >= window_s: wins.append(cur); cur, ws0 = [], None
    if cur: wins.append(cur)
    import bisect
    for tis in wins:
        a0, a1 = vv[tis[0]]["start"] - 15, vv[tis[-1]]["end"] + 15
        j0, j1 = bisect.bisect_left(pstart, a0), bisect.bisect_right(pstart, a1)
        s = set(tis)
        gk = [(i, k) for i, (ti, _, k) in enumerate(gv) if ti in s and k and k not in FILL] if len(wins) > 1 else [(i, k) for i, (ti, _, k) in enumerate(gv) if k and k not in FILL]
        pk = [(j, pn[j]) for j in range(j0, j1) if pn[j] and pn[j] not in FILL]
        sm = difflib.SequenceMatcher(None, [k for _, k in gk], [k for _, k in pk], autojunk=False)
        for blk in sm.get_matching_blocks():
            for q in range(blk.size):
                ti, wi, _ = gv[gk[blk.a + q][0]]; j = pk[blk.b + q][0]; t = vv[ti]
                if t["start"] - margin <= pw[j]["start"] and pw[j]["end"] <= t["end"] + margin: GA[(ti, wi)] = j
    byturn = {}
    for (ti, wi), j in GA.items(): byturn.setdefault(ti, {})[wi] = j
    turns, tot = [], dict(vv_words=0, matched=0, interpolated=0)
    for ti, t in enumerate(vv):
        toks = t["text"].split(); tn = [norm(x) for x in toks]; anch = dict(byturn.get(ti, {}))
        ka = [x if x and x not in FILL else None for x in tn]
        lo = min(anch.values(), default=0); hi = max(anch.values(), default=-1) + 1
        keys = sorted(anch)
        bounds = [(-1, lo - 1 if anch else -1)] + [(i, anch[i]) for i in keys] + [(len(tn), hi if anch else 0)]
        for (ia, ja), (ib, jb) in zip(bounds, bounds[1:]):  # fuzzy rescue of same-size gaps
            ai = [i for i in range(ia + 1, ib) if ka[i]]
            bj = [j for j in range(max(ja + 1, 0), min(jb, len(pw))) if pn[j] and pn[j] not in FILL and t["start"] - margin <= pw[j]["start"] and pw[j]["end"] <= t["end"] + margin]
            if len(ai) == len(bj) and ai:
                for i, j in zip(ai, bj):
                    if difflib.SequenceMatcher(None, tn[i], pn[j]).ratio() >= 0.6: anch[i] = j
        ws = [dict(w=x, start=None, end=None, matched=False) for x in toks]
        for i, j in anch.items(): ws[i].update(start=pw[j]["start"], end=pw[j]["end"], matched=True)
        n = len(ws); i = 0
        while i < n:  # interpolate runs of unmatched words
            if ws[i]["start"] is not None: i += 1; continue
            j = i
            while j < n and ws[j]["start"] is None: j += 1
            left = ws[i - 1]["end"] if i > 0 else None
            right = ws[j]["start"] if j < n else None
            if left is None: left = max(t["start"], right - 0.3 * (j - i)) if right is not None else t["start"]
            if right is None: right = min(t["end"], left + 0.3 * (j - i))
            right = max(right, left); step = (right - left) / (j - i)
            for k in range(i, j):
                ws[k]["start"] = round(left + step * (k - i), 3); ws[k]["end"] = round(left + step * (k - i + 1), 3)
            i = j
        nm = len(anch); tot["vv_words"] += n; tot["matched"] += nm; tot["interpolated"] += n - nm
        turns.append(dict(speaker=t["speaker"], start=ws[0]["start"] if ws else t["start"], end=ws[-1]["end"] if ws else t["end"],
                          vv_start=t["start"], vv_end=t["end"], text=t["text"], words=ws))
    return turns, tot


# ---------------------------------------------------------------- outputs
def hms(x): return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{x%60:04.1f}"
def srt_ts(x): x = max(x, 0); return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{int(x%60):02d},{int(round((x%1)*1000)):03d}".replace(",1000", ",999")


def split_turn(words, maxdur=6.0, maxw=14, gap=0.5):
    out, cur = [], []
    for w in words:
        if cur and (w["end"] - cur[0]["start"] > maxdur or len(cur) >= maxw or (w["start"] - cur[-1]["end"] > gap and len(cur) > 3)
                    or (cur[-1]["w"].endswith((".", "?", "!")) and len(cur) > 5)):
            out.append(cur); cur = []
        cur.append(w)
    if cur: out.append(cur)
    return out


def write_outputs(outdir, turns):
    spk = lambda t: f"SPEAKER_{t['speaker']:02d}"
    js = [dict(speaker=spk(t), **{k: v for k, v in t.items() if k != "speaker"}) for t in turns]
    json.dump(js, open(os.path.join(outdir, "transcript.json"), "w"), indent=1)
    with open(os.path.join(outdir, "transcript.txt"), "w") as f:
        for t in js: f.write(f"[{hms(t['start'])} - {hms(t['end'])}] {t['speaker']}: {t['text']}\n")
    n = 0
    with open(os.path.join(outdir, "subtitles.srt"), "w") as f:
        for t in js:
            words = t.get("words") or even_words(t)
            for c in split_turn(words):
                n += 1; f.write(f"{n}\n{srt_ts(c[0]['start'])} --> {srt_ts(c[-1]['end'])}\n{t['speaker']}: {' '.join(w['w'] for w in c)}\n\n")


def even_words(t):
    toks = t["text"].split(); d = (t["end"] - t["start"]) / max(len(toks), 1)
    return [dict(w=x, start=t["start"] + i * d, end=t["start"] + (i + 1) * d) for i, x in enumerate(toks)]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        a = sys.argv[2:]; return worker(a[0], float(a[1]), float(a[2]), a[3], a[4], int(a[5]), a[6])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="audio file, mp3 URL or RSS URL (first episode)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--chunk-minutes", type=float, default=50.0)
    ap.add_argument("--overlap-seconds", type=float, default=60.0)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--context", default="", help="hotwords / context hint passed to VibeVoice")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--no-words", action="store_true", help="skip Parakeet word timestamps")
    ap.add_argument("--keep-intermediate", action="store_true", help="keep work/ (wav, download)")
    args = ap.parse_args()
    out = os.path.abspath(args.outdir); work = os.path.join(out, "work"); os.makedirs(work, exist_ok=True)
    T = {}; t00 = time.time()
    t = time.time(); src = get_audio(args.input, work); wav = os.path.join(work, "audio_16k.wav"); dur = to_wav(src, wav); T["prepare_s"] = time.time() - t
    log(f"audio {dur/60:.1f} min -> {wav}")
    key = hashlib.sha1(open(wav, "rb").read()).hexdigest()[:10] + "_" + hashlib.sha1(f"{args.model}|{args.context}".encode()).hexdigest()[:6]
    plan = plan_chunks(dur, args.chunk_minutes * 60, args.overlap_seconds)
    log("chunk plan:", [(round(a), round(a + b)) for a, b in plan])
    t = time.time(); segs, infos = run_chunks(wav, plan, args, os.path.join(out, "chunks"), key); T["vibevoice_s"] = time.time() - t
    t = time.time(); vv, flips = stitch(segs, plan, args.overlap_seconds); T["stitch_s"] = time.time() - t
    log(f"stitched {len(vv)} turns; boundaries with raw speaker-label flip fixed: {sum(f['raw_labels_flipped'] for f in flips)}/{len(flips)}")
    stats = dict(input=args.input, model=args.model, audio_s=dur, chunk_plan=plan, chunks=infos, boundaries=flips, n_turns=len(vv),
                 speakers=sorted({s["speaker"] for s in vv}), vv_words=sum(len(s["text"].split()) for s in vv))
    if args.no_words:
        turns = vv
    else:
        t = time.time(); log("Parakeet word timestamps ..."); pw = parakeet_words(wav); T["parakeet_s"] = time.time() - t
        t = time.time(); turns, tot = align(vv, pw); T["align_s"] = time.time() - t
        stats.update(parakeet_words=len(pw), **tot, match_pct=round(100 * tot["matched"] / max(tot["vv_words"], 1), 2),
                     interpolated_pct=round(100 * tot["interpolated"] / max(tot["vv_words"], 1), 2))
        ws = [w for t_ in turns for w in t_["words"]]
        stats["non_monotonic_words"] = sum(1 for a, b in zip(ws, ws[1:]) if b["start"] < a["start"] - 1e-6)
        log(f"alignment: {stats['match_pct']}% matched, {stats['interpolated_pct']}% interpolated")
    write_outputs(out, turns)
    T["total_s"] = time.time() - t00
    gen = sum(c["gen_s"] for c in infos if not c.get("cached"))
    stats.update(timings_s={k: round(v, 1) for k, v in T.items()}, rtf_total=round(T["total_s"] / dur, 3),
                 vibevoice_gen_s_fresh=round(gen, 1), chunks_cached=sum(bool(c.get("cached")) for c in infos),
                 peak_mlx_gb=max(c["mlx_peak_gb"] for c in infos), peak_rss_gb_parent=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                 peak_rss_gb_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9)
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    if not args.keep_intermediate: shutil.rmtree(work, ignore_errors=True)
    log("done:", out, json.dumps(stats["timings_s"]))


if __name__ == "__main__":
    main()
