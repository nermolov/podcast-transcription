"""Scores bench.py runs: latency (stall-free time to playback) and quality vs the full-pass reference.

    <scratch venv with jiwer, sherpa-onnx>/bin/python research/streaming/analyze.py research/streaming/runs/<name> ... [--embed]

TTFP (time to first playback) = the smallest delay T0 such that, if playback starts at T0, every final segment is ready
before the playhead reaches its start: max_i(ready_i - start_i). Two readiness models from the same run:
  chunk:  a segment is ready when its whole chunk finished (what the current worker-per-chunk design gives)
  stream: a segment is ready when its JSON object closed in the token stream (and its chunk's speaker mapping is known)
Speaker mapping (chunk-local labels -> global ids): overlap (pipeline.stitch's greedy overlap matching), exemplar (roster clips
prepended at inference), embed (titanet speaker embeddings of each local label vs running global centroids).
"""
import argparse, json, os, re, sys, time, wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from transcriber.pipeline import align, FILL  # noqa: E402

REF = os.path.join(ROOT, "output/full_episode/transcript.json")
WAV = os.path.join(ROOT, "episode/full_16k.wav")
EMB_MODEL = os.path.join(ROOT, "research/runs/sherpa/models/nemo_en_titanet_small.onnx")


def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    chunks = []
    for k, (s0, nb, e0) in enumerate(meta["plan"]):
        p = os.path.join(d, f"chunk_{k}.json")
        if not os.path.exists(p): break
        c = json.load(open(p))
        emits = [e["t"] for e in ev if e["kind"] == "seg" and e["chunk"] == k]
        assert len(emits) == len(c["segs"]), (k, len(emits), len(c["segs"]))
        pre = c["prefix_s"]
        segs = []
        for s, t in zip(c["segs"], emits):
            if s.get("speaker_id") is None or not str(s.get("text", "")).strip(): continue
            segs.append(dict(lstart=float(s["start"]), lend=float(s["end"]), start=round(float(s["start"]) - pre + s0, 2), end=round(float(s["end"]) - pre + s0, 2),
                             local=int(s["speaker_id"]), text=str(s["text"]).strip(), emit=t, roster=float(s["end"]) <= pre + 0.25))
        end = next(e["t"] for e in ev if e["kind"] == "chunk_end" and e["chunk"] == k)
        # labels are known once the model has passed the overlap (and roster): first segment reaching the new region
        lab = next((s["emit"] for s in segs if s["end"] >= nb), end)
        chunks.append(dict(k=k, s0=s0, nb=nb, e0=e0, pre=pre, roster=c["roster"], segs=segs, end_t=end, label_t=lab))
    return meta, ev, chunks


# ---------------------------------------------------------------- speaker mapping
def map_overlap(chunks):
    """pipeline.stitch's greedy assignment by time overlap with the previous (already mapped) chunk."""
    maps, nxt = [], 100
    prev = None
    for c in chunks:
        body = [s for s in c["segs"] if not s["roster"]]
        if prev is None:
            mp = {s["local"]: s["local"] for s in body}
        else:
            a, b = c["s0"], prev["e0"]
            M = {}
            for s1 in prev["mapped"]:
                for s2 in body:
                    o = min(s1["end"], s2["end"], b) - max(s1["start"], s2["start"], a)
                    if o > 0: M[(s1["speaker"], s2["local"])] = M.get((s1["speaker"], s2["local"]), 0) + o
            mp, used = {}, set()
            for (p, l), o in sorted(M.items(), key=lambda x: -x[1]):
                if l not in mp and p not in used: mp[l] = p; used.add(p)
            prev_labels = {s["speaker"] for s in prev["mapped"]}
            for l in sorted({s["local"] for s in body}):
                if l not in mp:
                    free = sorted(prev_labels - used)
                    if free: mp[l] = free[0]; used.add(free[0])
                    else: mp[l] = nxt; nxt += 1
        c["mapped"] = [dict(s, speaker=mp[s["local"]]) for s in body]
        maps.append(mp); prev = c
    return maps


def map_exemplar(chunks):
    maps, nxt = [], 0
    for c in chunks:
        body = [s for s in c["segs"] if not s["roster"]]
        mp, used = {}, set()
        if c["roster"]:
            votes = {}
            for s in c["segs"]:
                for r in c["roster"]:
                    o = min(s["lend"], r["at"][1]) - max(s["lstart"], r["at"][0])
                    if o > 0: votes[(s["local"], r["g"])] = votes.get((s["local"], r["g"]), 0) + o
            for (l, g), o in sorted(votes.items(), key=lambda x: -x[1]):
                if l not in mp and g not in used: mp[l] = g; used.add(g)
        for l in sorted({s["local"] for s in body}):
            if l not in mp: mp[l] = nxt + 0; nxt += 1
        nxt = max(nxt, max(mp.values(), default=-1) + 1)
        c["mapped"] = [dict(s, speaker=mp[s["local"]]) for s in body]
        maps.append(mp)
    return maps


class Embedder:
    def __init__(self, wav=WAV):
        import sherpa_onnx
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=4))
        w = wave.open(wav); self.sr = w.getframerate(); self.pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.secs = 0.0

    def __call__(self, st, en, cap=10.0):
        t = time.time()
        st, en = st + 0.3, en - 0.3
        if en - st > cap: m = (st + en) / 2; st, en = m - cap / 2, m + cap / 2
        a = self.pcm[int(st * self.sr):int(en * self.sr)].astype(np.float32) / 32768
        s = self.ex.create_stream(); s.accept_waveform(self.sr, a); s.input_finished()
        e = np.array(self.ex.compute(s)); self.secs += time.time() - t
        return e / (np.linalg.norm(e) + 1e-9)


def map_embed(chunks, emb, thr=0.45, min_s=1.5):
    """Centroid per local label (segments >= min_s) vs running global centroids; Hungarian on cosine; below thr -> new speaker."""
    from scipy.optimize import linear_sum_assignment
    G = {}  # global id -> [sum vector, weight]
    maps, sims = [], []
    for c in chunks:
        body = [s for s in c["segs"] if not s["roster"]]
        loc = {}
        for s in body:
            d = s["end"] - s["start"]
            if d < min_s: continue
            v = emb(s["start"], s["end"]); w = min(d, 10.0)
            acc = loc.setdefault(s["local"], [np.zeros_like(v), 0.0]); acc[0] += w * v; acc[1] += w
        L = sorted(loc); gids = sorted(G); mp = {}
        if gids and L:
            S = np.array([[float(np.dot(loc[l][0] / np.linalg.norm(loc[l][0]), G[g][0] / np.linalg.norm(G[g][0]))) for g in gids] for l in L])
            r, cidx = linear_sum_assignment(-S)
            for i, j in zip(r, cidx):
                sims.append(round(S[i, j], 3))
                if S[i, j] >= thr: mp[L[i]] = gids[j]
        nxt = max(G, default=-1) + 1
        for l in sorted({s["local"] for s in body}):
            if l not in mp: mp[l] = nxt; nxt += 1
        for l, (v, w) in loc.items():
            acc = G.setdefault(mp[l], [np.zeros_like(v), 0.0]); acc[0] += v; acc[1] += w
        c["mapped"] = [dict(s, speaker=mp[s["local"]]) for s in body]
        maps.append(mp)
    return maps, sims


# ---------------------------------------------------------------- seams + metrics
def stitch_mapped(chunks, rule="v1"):
    """v1 = pipeline.stitch's seam rule on already-mapped chunks: cut each overlap at a segment end near its midpoint
    that is also a segment edge in the next chunk; keep chunk k up to the cut and chunk k+1 from it.
    It drops a whole segment when the next chunk opens with one long segment spanning the overlap (cut lands just
    after its start, and chunk k's copy ends past the cut). v2 cuts at a segment *start* of chunk k+1 inside the overlap
    (preferring one that matches a chunk-k segment end), keeps chunk k segments that start before it and chunk k+1
    segments from it on; with no start inside the overlap, it takes chunk k+1 from its first sample (duplicates over loss)."""
    if rule == "v2": return stitch_v2(chunks)
    allc = [c["mapped"] for c in chunks]
    cuts = []
    for k in range(len(allc) - 1):
        a, b = chunks[k + 1]["s0"], chunks[k]["e0"]; mid = (a + b) / 2
        edges = [x for s in allc[k + 1] for x in (s["start"], s["end"])]
        cands = [s["end"] for s in allc[k] if a + 1 <= s["end"] <= b and any(abs(s["end"] - e) <= 1.0 for e in edges)]
        if not cands: cands = [s["end"] for s in allc[k] if a <= s["end"] <= b] or [mid]
        cuts.append(min(cands, key=lambda x: abs(x - mid)))
    final = []
    for k, c in enumerate(allc):
        lo = cuts[k - 1] - 0.3 if k > 0 else -1; hi = cuts[k] + 1e-6 if k < len(cuts) else 1e12
        final += [dict(s, chunk=k) for s in c if s["start"] >= lo and s["end"] <= hi]
    final.sort(key=lambda s: s["start"])
    for x, y in zip(final, final[1:]):
        if y["start"] < x["end"]: x["end"] = y["start"]
    return final


def stitch_v2(chunks):
    allc = [c["mapped"] for c in chunks]
    cuts = []
    for k in range(len(allc) - 1):
        a, b = chunks[k + 1]["s0"], chunks[k]["e0"]; mid = (a + b) / 2
        starts = [s["start"] for s in allc[k + 1] if a + 1 <= s["start"] <= b - 1]
        good = [t for t in starts if any(abs(x["end"] - t) <= 1.0 for x in allc[k])]
        cands = good or starts
        if cands: cuts.append(min(cands, key=lambda x: abs(x - mid))); continue
        # chunk k+1 opens with one segment spanning the whole overlap: keep chunk k to its end and drop the words of that
        # segment that repeat chunk k's tail (last difflib match), so the seam neither loses nor duplicates text
        import difflib
        allc[k + 1] = [dict(s) for s in allc[k + 1]]
        b0 = allc[k + 1][0] if allc[k + 1] else None
        if b0 is None: cuts.append(b); continue
        tail = [w for s in allc[k] if s["end"] > a for w in s["text"].split()]
        bw = b0["text"].split()
        nw = lambda ws: [re.sub(r"[^a-z0-9']", "", w.lower()) for w in ws]
        sm = difflib.SequenceMatcher(None, nw(tail), nw(bw), autojunk=False)
        blocks = [m for m in sm.get_matching_blocks() if m.size >= 2]
        cut_i = blocks[-1].b + blocks[-1].size if blocks else 0
        b0["text"] = " ".join(bw[cut_i:]); b0["start"] = max(b0["start"], b)
        if not b0["text"]: allc[k + 1] = allc[k + 1][1:]
        cuts.append(b + 0.5)  # chunk k keeps everything starting before its end; chunk k+1 from the trimmed b0
    final = []
    for k, c in enumerate(allc):
        lo = cuts[k - 1] - 0.5 if k > 0 else -1; hi = cuts[k] - 0.5 if k < len(cuts) else 1e12
        final += [dict(s, chunk=k) for s in c if lo <= s["start"] < hi]
    final.sort(key=lambda s: s["start"])
    for x, y in zip(final, final[1:]):
        if y["start"] < x["end"]: x["end"] = y["start"]
    return final


def ttfp(final, chunks):
    ch = {c["k"]: c for c in chunks}
    stream = max(max(s["emit"], ch[s["chunk"]]["label_t"]) - s["start"] for s in final)
    chunk = max(ch[s["chunk"]]["end_t"] - s["start"] for s in final)
    worst = max(final, key=lambda s: max(s["emit"], ch[s["chunk"]]["label_t"]) - s["start"])
    # headroom per chunk if playback starts at T0 = stream TTFP: min over its segments of (playhead reaches start) - ready
    head = [round(min((stream + s["start"] - max(s["emit"], c["label_t"]) for s in final if s["chunk"] == c["k"]), default=0), 1) for c in chunks]
    return round(max(0.0, stream), 1), round(max(0.0, chunk), 1), dict(at=worst["start"], chunk=worst["chunk"], stream_headroom_s=head)


def grid_labels(segs, dur, step=0.25):
    lab = np.full(int(dur / step) + 1, -1)
    for s in segs:
        lab[int(s["start"] / step):int(s["end"] / step)] = s["speaker"] if isinstance(s["speaker"], int) else int(str(s["speaker"]).split("_")[-1])
    return lab


def speaker_agreement(final, ref, dur, lo=0.0, hi=None):
    """Time-weighted label agreement with the reference after the best global hyp->ref relabeling."""
    from scipy.optimize import linear_sum_assignment
    h, r = grid_labels(final, dur), grid_labels(ref, dur)
    sl = slice(int(lo / 0.25), int((hi or dur) / 0.25))
    h, r = h[sl], r[sl]
    m = (h >= 0) & (r >= 0)
    if not m.any(): return float("nan"), {}  # no reference in this range
    hs, rs = sorted(set(h[m])), sorted(set(r[m]))
    C = np.array([[np.sum((h[m] == a) & (r[m] == b)) for b in rs] for a in hs])
    i, j = linear_sum_assignment(-C)
    return round(100 * C[i, j].sum() / max(m.sum(), 1), 2), {int(hs[a]): int(rs[b]) for a, b in zip(i, j)}


def norm_words(t):
    return [w for w in (re.sub(r"[^a-z0-9']", "", x.lower().replace("’", "'")) for x in t.split()) if w and w not in FILL]


def wer(ref_segs, hyp_segs, lo=0.0, hi=1e9):
    import jiwer
    r = " ".join(w for s in ref_segs if lo <= s["start"] < hi for w in norm_words(s["text"]))
    h = " ".join(w for s in hyp_segs if lo <= s["start"] < hi for w in norm_words(s["text"]))
    o = jiwer.process_words(r, h)
    return dict(wer=round(100 * o.wer, 2), sub=o.substitutions, ins=o.insertions, dele=o.deletions, ref_words=len(r.split()), hyp_words=len(h.split()))


def pk_words(d, n):
    words = []
    for k in range(n):
        p = os.path.join(d, f"pk_tokens_{k}.json")
        if not os.path.exists(p): return None
        cur = None
        for tok in json.load(open(p)):
            if cur is None or tok["w"].startswith(" "):
                if cur: words.append(cur)
                cur = dict(w=tok["w"].strip(), start=tok["start"], end=tok["end"])
            else: cur["w"] += tok["w"]; cur["end"] = tok["end"]
        if cur: words.append(cur)
    return [w for w in words if w["w"]]


def analyze(d, embedder=None, ref_path=REF):
    meta, ev, chunks = load_run(d)
    dur = meta["audio_s"]; done = len(chunks) == len(meta["plan"])
    ref = json.load(open(ref_path))
    hi = chunks[-1]["e0"] if chunks else 0
    t = {e["kind"]: e["t"] for e in ev if e["kind"] in ("prepared", "loaded", "done")}
    res = dict(run=os.path.basename(d.rstrip("/")), args=meta["args"]["plan"], speaker_mode=meta["args"]["speaker"], overlap=meta["args"]["overlap"],
               complete=done, n_chunks=len(chunks), prepared_t=t.get("prepared"), loaded_t=t.get("loaded"), total_t=t.get("done"),
               chunks=[dict(k=c["k"], audio=[c["s0"], c["e0"]], end_t=c["end_t"], label_t=c["label_t"],
                            secs=next(e["secs"] for e in ev if e["kind"] == "chunk_end" and e["chunk"] == c["k"]),
                            first_token_s=next(e["secs"] for e in ev if e["kind"] == "first_token" and e["chunk"] == c["k"]),
                            words_s=next((e["secs"] for e in ev if e["kind"] == "words" and e["chunk"] == c["k"]), 0),
                            peak_gb=next(e["mlx_peak_gb"] for e in ev if e["kind"] == "chunk_end" and e["chunk"] == c["k"])) for c in chunks],
               mappings={})
    modes = ["overlap"] + (["exemplar"] if meta["args"]["speaker"] == "exemplar" else []) + (["embed"] if embedder else [])
    for mode in modes:
        if mode == "overlap": maps = map_overlap(chunks); extra = {}
        elif mode == "exemplar": maps = map_exemplar(chunks); extra = {}
        else:
            s0 = embedder.secs; maps, sims = map_embed(chunks, embedder)
            extra = dict(embed_cpu_s=round(embedder.secs - s0, 1), matched_cos=sims)
        final = stitch_mapped(chunks)
        st, ck, worst = ttfp(final, chunks)
        agree, rel = speaker_agreement(final, ref, dur, hi=hi)
        r = dict(ttfp_stream_s=st, ttfp_chunk_s=ck, worst=worst, first_text_t=final[0]["emit"], speaker_agree_pct=agree, relabel=rel,
                 n_speakers=len({s["speaker"] for s in final}), maps=[{str(a): b for a, b in m.items()} for m in maps],
                 # per-chunk agreement localizes a bad seam
                 chunk_agree=[speaker_agreement(final, ref, dur, lo=c["nb"], hi=c["e0"])[0] for c in chunks], **extra)
        if mode == "overlap":
            r["text_vs_ref"] = wer(ref, final, 0, hi)
            f2 = stitch_mapped(chunks, "v2")
            r["seam_v2"] = dict(text_vs_ref=wer(ref, f2, 0, hi), speaker_agree_pct=speaker_agreement(f2, ref, dur, hi=hi)[0], ttfp=ttfp(f2, chunks)[:2],
                                gaps_over_4s=sum(y["start"] - x["end"] > 4 for x, y in zip(f2, f2[1:])))
            r["gaps_over_4s"] = sum(y["start"] - x["end"] > 4 for x, y in zip(final, final[1:]))
            pw = pk_words(d, len(chunks))
            if pw:
                turns, tot = align([dict(start=s["start"], end=s["end"], speaker=s["speaker"], text=s["text"]) for s in final], pw)
                r["word_match_pct"] = round(100 * tot["matched"] / max(tot["vv_words"], 1), 2)
            json.dump([{k: s[k] for k in ("start", "end", "speaker", "text", "emit", "chunk")} for s in final], open(os.path.join(d, "final_overlap.json"), "w"), indent=1)
        res["mappings"][mode] = r
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--embed", action="store_true")
    ap.add_argument("--ref", default=REF, help="reference transcript.json"); ap.add_argument("--wav", default=WAV, help="16 kHz mono wav (for --embed)")
    a = ap.parse_args()
    emb = Embedder(a.wav) if a.embed else None
    for d in a.runs:
        r = analyze(d, emb, a.ref)
        json.dump(r, open(os.path.join(d, "summary.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in r.items() if k != "chunks"}, indent=1))
