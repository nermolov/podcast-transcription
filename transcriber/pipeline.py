"""Podcast -> speaker-attributed transcript with word timestamps (VibeVoice-ASR + Parakeet, MLX)."""
import atexit, difflib, hashlib, json, math, os, re, resource, shutil, signal, subprocess, sys, threading, time, urllib.request, wave

MODEL_DEFAULT = "mlx-community/VibeVoice-ASR-8bit"
PIPELINE_VERSION = 2
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v3"
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILL = {"uh", "um", "uhm", "mm", "hmm", "mhm", "erm", "ah"}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


class TranscribeError(Exception):
    """Expected failure (bad input, ffmpeg/worker failure); the CLI turns it into an exit message."""


# ---------------------------------------------------------------- child process management
_CHILDREN = set(); _CH_LOCK = threading.Lock()


def kill_children(grace=5.0):
    """SIGTERM (then SIGKILL after `grace` s) every live child started by run_child, including its process group."""
    with _CH_LOCK: procs = list(_CHILDREN)
    def sig(p, s):
        try: os.killpg(p.pid, s)  # children lead their own session/group (start_new_session)
        except (ProcessLookupError, PermissionError): pass
    for p in procs: sig(p, signal.SIGTERM)
    end = time.time() + grace
    for p in procs:
        try: p.wait(max(0.0, end - time.time()))
        except subprocess.TimeoutExpired: sig(p, signal.SIGKILL); p.wait()


atexit.register(kill_children, 2.0)


def install_signal_handlers():
    """For standalone use (CLI): on SIGTERM/SIGINT/SIGHUP kill tracked children, then die with the default action.
    Main thread only. (uvicorn installs its own handlers; the server cleans up in its shutdown path + atexit.)"""
    def h(signum, frame):
        kill_children()
        signal.signal(signum, signal.SIG_DFL); os.kill(os.getpid(), signum)
    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sg, h)


def run_child(cmd, env=None):
    """subprocess.run(check=False) that tracks the child so kill_children()/atexit can stop it; returns the exit code.
    The child is in its own session, so a terminal Ctrl-C does not reach it directly (the parent's handler kills it)."""
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    with _CH_LOCK: _CHILDREN.add(p)
    try: return p.wait()
    except BaseException: kill_children(); raise
    finally:
        with _CH_LOCK: _CHILDREN.discard(p)


# ---------------------------------------------------------------- input handling
def get_audio(src, work):
    if re.match(r"https?://", src):
        req = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}))
        r = req(src); head = r.read(512)
        if head.lstrip().startswith((b"<?xml", b"<rss", b"<feed")):
            xml = head + r.read()
            m = re.search(rb'<enclosure[^>]+url=["\']([^"\']+)', xml)
            if not m: raise TranscribeError("RSS feed has no <enclosure> url")
            src = m.group(1).decode().replace("&amp;", "&"); log("RSS: first episode ->", src); r = req(src); head = r.read(512)
        dst = os.path.join(work, "download.audio")
        with open(dst, "wb") as f:
            f.write(head); shutil.copyfileobj(r, f)
        return dst
    if not os.path.exists(src): raise TranscribeError(f"not found: {src}")
    return src


def to_wav(src, wav):
    try: subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav], check=True)
    except FileNotFoundError: raise TranscribeError("ffmpeg not found on PATH")
    except subprocess.CalledProcessError as e: raise TranscribeError(f"ffmpeg failed (exit {e.returncode}) on {src}")
    w = wave.open(wav); return w.getnframes() / w.getframerate()


def plan_chunks(dur, chunk_s, ov):
    if dur <= chunk_s: return [(0.0, dur)]
    n = math.ceil((dur - ov) / (chunk_s - ov))
    L = (dur + (n - 1) * ov) / n
    return [(round(k * (L - ov), 2), round(min(L, dur - k * (L - ov)), 2)) for k in range(n)]




PROGRESSIVE = (120.0, 360.0, 1200.0)  # new audio (s) in the first chunks; later chunks are chunk_minutes long
MAX_CHUNK_S = 59 * 60  # mlx-audio's VibeVoice limit per pass


def progressive_plan(dur, chunk_s, ov, lens=PROGRESSIVE):
    """[(audio_start, new_start, end)] with new audio growing 120 s -> 360 s -> 1200 s -> chunk_s; chunk k decodes
    [new_start - ov, end]. Small first chunks start playback early; each one only has to finish before the playhead reaches
    the next, so ~3x growth keeps tens of seconds of headroom (see research/streaming/RESULTS.md). A short tail is folded
    into the previous chunk."""
    lens = [min(x, chunk_s) for x in lens] + [chunk_s]
    bounds, b, k = [0.0], 0.0, 0
    while b < dur - 1e-6:
        L = lens[min(k, len(lens) - 1)]
        if dur - (b + L) < max(30.0, 0.15 * L) and dur - b + ov <= MAX_CHUNK_S: L = dur - b
        b = min(dur, b + L); bounds.append(round(b, 2)); k += 1
    return [(max(0.0, round(bounds[i] - (ov if i else 0), 2)), bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def fixed_plan(dur, chunk_s, ov):
    """plan_chunks (equal windows) in progressive_plan's (audio_start, new_start, end) form."""
    p = plan_chunks(dur, chunk_s, ov)
    return [(st, st if k == 0 else round(p[k - 1][0] + p[k - 1][1], 2), round(st + du, 2)) for k, (st, du) in enumerate(p)]


def stream_child(cmd, env=None):
    """Like run_child, but yields the child's stdout lines parsed as JSON; raises TranscribeError on a nonzero exit."""
    p = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=subprocess.PIPE, text=True, bufsize=1)
    with _CH_LOCK: _CHILDREN.add(p)
    try:
        for line in p.stdout:
            line = line.strip()
            if line.startswith("{"): yield json.loads(line)
        rc = p.wait()
        if rc != 0: raise TranscribeError(f"VibeVoice worker failed (exit {rc})")
    except BaseException:
        kill_children(); raise
    finally:
        with _CH_LOCK: _CHILDREN.discard(p)


# ---------------------------------------------------------------- speaker mapping (chunk-local labels -> global ids)
def overlap_votes(prev, cur, a, b):
    """{(global id in chunk k-1, local label in chunk k): seconds both say were spoken} over the overlap [a, b]."""
    M = {}
    for s1 in prev:
        for s2 in cur:
            o = min(s1["end"], s2["end"], b) - max(s1["start"], s2["start"], a)
            if o > 0: M[(s1["speaker"], s2["local"])] = M.get((s1["speaker"], s2["local"]), 0) + o
    return M


def greedy(scores):
    """Greedy one-to-one assignment over {(x, y): score}, best first -> {y: x}."""
    mp, used = {}, set()
    for (x, y), _ in sorted(scores.items(), key=lambda kv: -kv[1]):
        if y not in mp and x not in used: mp[y] = x; used.add(x)
    return mp


class OverlapMapper:
    """Match labels by how long both chunks agree someone spoke in the overlap. A label with no overlap evidence takes a
    previous-chunk speaker not used in this chunk (correct for two speakers only), else a new id."""
    name = "overlap"

    def __init__(self): self.nxt = 0

    def decide(self, k, prev, cur, a, b):
        M = overlap_votes(prev, cur, a, b)
        mp = greedy(M)
        return mp, dict(overlap_s=round(sum(M.values()), 1), overlap_agree_s=round(sum(o for (g, l), o in M.items() if mp.get(l) == g), 1))

    def decide_late(self, k, segs, used, prev, force):
        """Global id for a label of chunk k not mapped by decide() (`segs`: its segments so far), or None to wait for more of it."""
        free = sorted({s["speaker"] for s in prev} - set(used))
        return free[0] if free else self.new()

    def new(self): self.nxt += 1; return self.nxt - 1

    def chunk_end(self, k, segs, mp):
        self.nxt = max([self.nxt] + [g + 1 for g in mp.values()])


class EmbedMapper(OverlapMapper):
    """Match labels by speaker embedding: each local label's centroid (segments >= min_s, middle <= 10 s) against running
    global centroids, one-to-one by best cosine; below thr it is a new speaker. Independent of who talks in the overlap.
    The overlap mapping is still computed and disagreements are recorded in the boundary info."""
    name = "embed"

    def __init__(self, embed, thr=0.45, late_thr=0.35, min_s=1.5):
        super().__init__(); self.embed = embed; self.thr = thr; self.late_thr = late_thr; self.min_s = min_s
        self.G = {}  # global id -> [weighted sum of unit vectors, weight]
        self.cache = {}

    def vec(self, s, min_s=None):
        if s["end"] - s["start"] < (self.min_s if min_s is None else min_s): return None
        key = (s["start"], s["end"])
        if key not in self.cache: self.cache[key] = self.embed(s["start"], s["end"])
        return self.cache[key]

    def centroid(self, segs):
        import numpy as np
        acc = None
        for s in segs:
            v = self.vec(s)
            if v is None: continue
            w = min(s["end"] - s["start"], 10.0); acc = w * v if acc is None else acc + w * v
        return None if acc is None else acc / (np.linalg.norm(acc) + 1e-9)

    def gvec(self, g):
        import numpy as np
        v = self.G[g][0]; return v / (np.linalg.norm(v) + 1e-9)

    def decide(self, k, prev, cur, a, b):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        ov, info = OverlapMapper.decide(self, k, prev, cur, a, b)
        L = [l for l in sorted({s["local"] for s in cur}) if self.centroid([s for s in cur if s["local"] == l]) is not None]
        gids = sorted(self.G); mp, sims = {}, {}
        if L and gids:
            C = {l: self.centroid([s for s in cur if s["local"] == l]) for l in L}
            S = np.array([[float(C[l] @ self.gvec(g)) for g in gids] for l in L])
            for i, j in zip(*linear_sum_assignment(-S)):
                sims[L[i]] = round(S[i, j], 3)
                if S[i, j] >= self.thr: mp[L[i]] = gids[j]
        # labels with too little speech to embed yet (backchannels) take the overlap match when it does not conflict
        via = {l: g for l, g in ov.items() if l not in mp and l not in L and g not in mp.values()}
        mp.update(via)
        info.update(cos=sims, overlap_map=ov, via_overlap=sorted(via), overlap_disagrees=sorted(l for l in ov if l in mp and ov[l] != mp[l]))
        return mp, info

    def decide_late(self, k, segs, used, prev, force):
        """With >= min_s of usable speech: best unused global speaker if cosine >= late_thr; a new speaker only once >= 3 s of it
        matches nobody. Labels with only short segments wait (until `force`, at chunk end) and are never made new speakers
        while known ones exist: a backchannel is far more likely a known voice than a new one."""
        c = self.centroid(segs)
        cand = [g for g in self.G if g not in used] or list(self.G)
        if c is not None and cand:
            best = max((float(c @ self.gvec(g)), g) for g in cand)
            if best[0] >= self.late_thr: return best[1]
            if sum(s["end"] - s["start"] for s in segs if s["end"] - s["start"] >= self.min_s) >= 3.0 or force: return self.new()
            return None
        if not force: return None
        if not cand: return self.new()
        v = self.vec(max(segs, key=lambda s: s["end"] - s["start"]), min_s=0.5)
        if v is not None: return max((float(v @ self.gvec(g)), g) for g in cand)[1]
        return OverlapMapper.decide_late(self, k, segs, used, prev, force) if prev else cand[0]

    def chunk_end(self, k, segs, mp):
        super().chunk_end(k, segs, mp)
        for s in segs:
            v = self.vec(s)
            if v is None or s["local"] not in mp: continue
            w = min(s["end"] - s["start"], 10.0); acc = self.G.setdefault(mp[s["local"]], [0.0, 0.0])
            acc[0] = acc[0] + w * v; acc[1] += w


class Embedder:
    """Unit speaker embedding of wav[start:end] (titanet-small via sherpa-onnx, CPU), middle `cap` seconds, 0.3 s trimmed at each end."""
    def __init__(self, wav, model_path=None, threads=2):
        import numpy as np, sherpa_onnx
        self.np = np
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path or speaker_model_path(), num_threads=threads))
        w = wave.open(wav); self.sr = w.getframerate(); self.pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    def __call__(self, st, en, cap=10.0):
        np = self.np
        st, en = st + 0.3, en - 0.3
        if en - st < 0.3: st, en = st - 0.3, en + 0.3
        if en - st > cap: m = (st + en) / 2; st, en = m - cap / 2, m + cap / 2
        s = self.ex.create_stream(); s.accept_waveform(self.sr, self.pcm[int(st * self.sr):int(en * self.sr)].astype(np.float32) / 32768); s.input_finished()
        e = np.array(self.ex.compute(s)); return e / (np.linalg.norm(e) + 1e-9)


SPEAKER_MODEL = ("csukuangfj/speaker-embedding-models", "nemo_en_titanet_small.onnx")


def speaker_model_path():
    from huggingface_hub import hf_hub_download
    try: return hf_hub_download(*SPEAKER_MODEL, local_files_only=True)  # no network when cached (fetch_models)
    except Exception: return hf_hub_download(*SPEAKER_MODEL)


def make_mapper(speakers, wav):
    if speakers == "overlap": return OverlapMapper()
    if speakers == "embed": return EmbedMapper(Embedder(wav))
    raise TranscribeError(f"unknown speaker mapping {speakers!r}")


# ---------------------------------------------------------------- incremental stitching
class Stitcher:
    """Stitches streamed chunk segments into one transcript with global speaker ids, emitting each segment as soon as it is final.

    add(k, seg) (seg times relative to the chunk's audio start), chunk_done(k) and finish() return the newly final segments
    (absolute times, global `speaker`), in time order; a final segment never changes afterwards, and `ready_until` is the
    audio time up to which the transcript is final. Chunks arrive in order (one worker).
    - Labels of chunk k > 0 are mapped once its output has passed the overlap (first segment ending at/after new_start), with the
      overlap and the previous chunk complete; a label first seen later goes to mapper.decide_late, which may wait for more of its
      speech (finalization pauses meanwhile). Chunk 0's labels get new ids in order of appearance.
    - Seam k-1|k is decided at the same moment: cut at a chunk-k segment *start* inside the overlap (preferring one within 1 s of
      a chunk k-1 segment end, nearest the overlap middle); chunk k-1 keeps segments starting before the cut, chunk k from it.
      If no chunk-k segment starts inside the overlap (one long segment spans it), chunk k-1 is kept to its end and the words of
      that segment repeating chunk k-1's tail are removed (last difflib match of >= 2 words), so text is neither lost nor doubled.
    - Segments of chunk k starting before chunk k+1's audio are final as soon as their label is mapped; later ones wait for the seam.
    """
    def __init__(self, plan, mapper, max_wait_s=30.0):
        self.plan, self.mapper, self.max_wait_s = plan, mapper, max_wait_s
        self.raw = [[] for _ in plan]; self.maps = [None] * len(plan); self.cuts = [None] * len(plan); self.done = [False] * len(plan)
        self.final, self.boundaries = [], []
        self.fk, self.fi = 0, 0  # next raw segment to finalize
        self.maps[0] = {}

    @property
    def ready_until(self):
        if self.fk >= len(self.plan): return self.plan[-1][2]
        return self.final[-1]["end"] if self.final else 0.0

    def add(self, k, seg):
        s0, nb, e0 = self.plan[k]
        if seg.get("speaker_id") is None or not str(seg.get("text", "")).strip(): return []
        s = dict(start=round(float(seg["start"]) + s0, 2), end=round(float(seg["end"]) + s0, 2), local=int(seg["speaker_id"]), text=str(seg["text"]).strip())
        self.raw[k].append(s)
        if k > 0 and self.maps[k] is None and s["end"] >= nb: self._decide(k)
        return self._advance()

    def chunk_done(self, k):
        self.done[k] = True
        if k > 0 and self.maps[k] is None: self._decide(k)
        for s in self.raw[k]:  # every label of a finished chunk is mapped (the next chunk's overlap matching needs them)
            if s["local"] not in self.maps[k]: self.maps[k][s["local"]] = self._late(k, s["local"], force=True)
        self.mapper.chunk_end(k, self.raw[k], self.maps[k])
        return self._advance()

    def finish(self):
        for k in range(len(self.plan)):
            if not self.done[k]: self.chunk_done(k)
        return self._advance()

    def _late(self, k, label, force=False):
        if k == 0: return self.mapper.new()
        segs = [s for s in self.raw[k] if s["local"] == label]
        # waiting stalls finalization: after max_wait_s of later audio, decide with what there is
        force = force or self.done[k] or self.raw[k][-1]["end"] - segs[0]["start"] > self.max_wait_s
        return self.mapper.decide_late(k, segs, set(self.maps[k].values()), self._mapped(k - 1), force)

    def _mapped(self, k):
        return [dict(s, speaker=self.maps[k][s["local"]]) for s in self.raw[k] if s["local"] in self.maps[k]]

    def _decide(self, k):
        a, b = self.plan[k][0], self.plan[k - 1][2]
        prev, cur = self._mapped(k - 1), self.raw[k]
        mp, info = self.mapper.decide(k, prev, cur, a, b)
        self.maps[k] = mp
        info.update(boundary=k, overlap=[a, b], raw_labels_flipped=any(l != g for l, g in mp.items()), mapping=dict(mp))
        # seam k-1|k
        mid = (a + b) / 2
        starts = [s["start"] for s in cur if a + 1 <= s["start"] <= b - 1]
        good = [t for t in starts if any(abs(x["end"] - t) <= 1.0 for x in self.raw[k - 1])]
        cands = good or starts
        if cands:
            self.cuts[k - 1] = min(cands, key=lambda x: abs(x - mid)); info["seam"] = "start" if good else "start-unmatched"
        else:
            self.cuts[k - 1] = b + 0.5; info["seam"] = "dedupe"
            if cur:
                tail = [w for x in self.raw[k - 1] if x["end"] > a for w in x["text"].split()]
                bw = cur[0]["text"].split()
                sm = difflib.SequenceMatcher(None, [norm(w) for w in tail], [norm(w) for w in bw], autojunk=False)
                blocks = [m for m in sm.get_matching_blocks() if m.size >= 2]
                cut_i = blocks[-1].b + blocks[-1].size if blocks else 0
                cur[0]["text"] = " ".join(bw[cut_i:]); cur[0]["start"] = max(cur[0]["start"], b); info["dedupe_words"] = cut_i
                if not cur[0]["text"]: cur.pop(0)
        info["cut"] = self.cuts[k - 1]
        self.boundaries.append(info)

    def _advance(self):
        out = []
        while self.fk < len(self.plan):
            k = self.fk; segs = self.raw[k]
            if self.maps[k] is None: break
            if self.fi >= len(segs):
                if self.done[k] and (k == len(self.plan) - 1 or self.cuts[k] is not None):
                    self.fk += 1; self.fi = 0; continue
                break
            s = segs[self.fi]
            if k > 0 and s["start"] < self.cuts[k - 1] - 0.5: self.fi += 1; continue  # before the seam: chunk k-1's copy is used
            if k < len(self.plan) - 1 and s["start"] >= self.plan[k + 1][0]:
                if self.cuts[k] is None: break  # inside the next overlap: wait for the seam
                if s["start"] >= self.cuts[k] - 0.5: self.fi = len(segs); continue  # after the seam: chunk k+1's copy is used
            if s["local"] not in self.maps[k]:
                g = self._late(k, s["local"])
                if g is None: break  # label not identifiable yet: wait for more of its speech (or the chunk end)
                self.maps[k][s["local"]] = g
            f = dict(start=s["start"], end=s["end"], speaker=self.maps[k][s["local"]], text=s["text"], chunk=k)
            if self.final and f["start"] < self.final[-1]["end"]: f["start"] = min(self.final[-1]["end"], f["end"])
            self.final.append(f); out.append(f); self.fi += 1
        return out


def stitch(chunk_segs, plan, ov=None):
    """Batch form of Stitcher with overlap matching, for (start, dur) plans: returns (segments, boundary info)."""
    p = [(st, st if k == 0 else plan[k - 1][0] + plan[k - 1][1], st + du) for k, (st, du) in enumerate(plan)]
    st = Stitcher(p, OverlapMapper())
    for k, segs in enumerate(chunk_segs):
        for s in segs: st.add(k, s)
        st.chunk_done(k)
    st.finish()
    return [{x: s[x] for x in ("start", "end", "speaker", "text")} for s in st.final], st.boundaries


# ---------------------------------------------------------------- word alignment
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
    js = [turn_json(t) for t in turns]
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




def turn_json(t):
    return dict(speaker=f"SPEAKER_{t['speaker']:02d}", **{k: v for k, v in t.items() if k not in ("speaker", "chunk")})


def write_partial(outdir, turns, ready_until, dur, complete=False):
    """partial.json: the final part of the transcript so far (rewritten atomically as it grows); `ready_until` is the audio time
    up to which it is final, so a player can start once it covers the first turn and keep going while it stays ahead."""
    p = os.path.join(outdir, "partial.json")
    json.dump(dict(complete=complete, audio_s=dur, ready_until=round(ready_until, 2), speakers=sorted({f"SPEAKER_{t['speaker']:02d}" for t in turns}),
                   turns=[turn_json(t) for t in turns]), open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


def run(input, outdir, *, chunk_minutes=40, overlap_seconds=30, progressive=True, speakers="embed", model=MODEL_DEFAULT, context="",
        max_tokens=40000, words=True, keep_intermediate=False, progress=None, input_name=None, cache_dir=None):
    """Transcribe `input` (file, mp3 URL or RSS URL) into `outdir`; returns the stats dict (also written to stats.json).
    One worker process runs the chunk plan (progressive_plan, or equal fixed_plan windows if progressive=False), streaming segments;
    they are stitched and speaker-mapped (`speakers`: "embed" or "overlap") as they arrive, word-aligned, and appended to
    <outdir>/partial.json as soon as they are final. `cache_dir` holds the per-chunk results (default <outdir>/chunks; the server
    shares one per audio sha256). `input_name` overrides the `input` recorded in stats.json (e.g. original upload name).
    progress(stage, fraction|None, message, **info) is called for stages prepare, vibevoice (fraction = final audio / duration;
    info ready_s = audio time up to which the transcript is final, audio_s = duration), write."""
    prog = progress or (lambda *a, **k: None)
    out = os.path.abspath(outdir); work = os.path.join(out, "work"); os.makedirs(work, exist_ok=True)
    T = {}; t00 = time.time()
    prog("prepare", None, "fetching and converting audio")
    t = time.time(); src = get_audio(input, work); wav = os.path.join(work, "audio_16k.wav"); dur = to_wav(src, wav); T["prepare_s"] = time.time() - t
    log(f"audio {dur/60:.1f} min -> {wav}")
    key = hashlib.sha1(open(wav, "rb").read()).hexdigest()[:10] + "_" + hashlib.sha1(f"{model}|{context}".encode()).hexdigest()[:6]
    plan = (progressive_plan if progressive else fixed_plan)(dur, chunk_minutes * 60, overlap_seconds)
    log("chunk plan:", [(round(a), round(b), round(c)) for a, b, c in plan])
    st = Stitcher(plan, make_mapper(speakers, wav))
    pw, turns, infos, tot = [], [], [], dict(vv_words=0, matched=0, interpolated=0)
    ready_log = []  # (seconds since start, turn start) per final turn -> time-to-playback stats
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [PKG_ROOT, os.environ.get("PYTHONPATH")])))
    cmd = [sys.executable, "-m", "transcriber.worker", wav, json.dumps(plan), model, context or "", str(max_tokens),
           cache_dir or os.path.join(out, "chunks"), key, "1" if words else "0"]
    t = time.time()
    for ev in stream_child(cmd, env):
        new = []
        if ev["ev"] == "words": pw += ev["words"]
        elif ev["ev"] == "seg": new = st.add(ev["k"], ev["seg"])
        elif ev["ev"] == "chunk_start": log(f"chunk {ev['k']+1}/{len(plan)} {plan[ev['k']]}: {'cached' if ev['cached'] else 'running'}")
        elif ev["ev"] == "chunk_end":
            new = st.chunk_done(ev["k"]); s = ev["stats"]; infos.append(s)
            if not s.get("cached"):
                log(f"chunk {ev['k']+1}/{len(plan)} done: words {s['words_s']:.0f}s, VibeVoice {s['gen_s']:.0f}s, RTF {s['rtf']:.3f}, {s['gen_tokens']} tokens, MLX peak {s['mlx_peak_gb']:.1f} GB")
                if s["hit_max_tokens"]: log("WARNING: chunk hit max_tokens, output truncated; raise --max-tokens")
        if new:
            turns += add_words(new, pw, tot) if words else [dict(x) for x in new]
            now = time.time() - t00; ready_log += [(now, x["start"]) for x in new]
            write_partial(out, turns, st.ready_until, dur)
            prog("vibevoice", min(st.ready_until / dur, 1.0), f"final up to {hms(st.ready_until)}", ready_s=st.ready_until, audio_s=dur)
    new = st.finish()
    turns += add_words(new, pw, tot) if words and new else [dict(x) for x in new]
    T["transcribe_s"] = time.time() - t
    log(f"{len(turns)} turns; seams: {[b.get('seam') for b in st.boundaries]}; raw label flips fixed: {sum(b['raw_labels_flipped'] for b in st.boundaries)}/{len(st.boundaries)}")
    stats = dict(input=input_name or input, model=model, audio_s=dur, progressive=progressive, speakers_mapping=speakers, chunk_plan=plan, chunks=infos,
                 boundaries=st.boundaries, n_turns=len(turns), speakers=sorted({x["speaker"] for x in turns}), vv_words=sum(len(x["text"].split()) for x in turns),
                 first_final_s=round(ready_log[0][0], 1) if ready_log else None,
                 # earliest playback start that never catches up with the transcript (for this run's speed)
                 time_to_playback_s=round(max([0.0] + [tt - s for tt, s in ready_log]), 1))
    if words:
        ws = [w for x in turns for w in x["words"]]
        stats.update(parakeet_words=len(pw), **tot, match_pct=round(100 * tot["matched"] / max(tot["vv_words"], 1), 2),
                     interpolated_pct=round(100 * tot["interpolated"] / max(tot["vv_words"], 1), 2),
                     non_monotonic_words=sum(1 for a, b in zip(ws, ws[1:]) if b["start"] < a["start"] - 1e-6))
        log(f"alignment: {stats['match_pct']}% matched, {stats['interpolated_pct']}% interpolated")
    prog("write", None, "writing outputs")
    write_outputs(out, turns)
    write_partial(out, turns, dur, dur, complete=True)
    T["total_s"] = time.time() - t00
    fresh = [c for c in infos if not c.get("cached")]
    stats.update(timings_s={k: round(v, 1) for k, v in T.items()}, rtf_total=round(T["total_s"] / dur, 3),
                 vibevoice_gen_s_fresh=round(sum(c["gen_s"] for c in fresh), 1), chunks_cached=len(infos) - len(fresh),
                 peak_mlx_gb=max([c["mlx_peak_gb"] for c in infos] or [0]), peak_rss_gb_parent=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                 peak_rss_gb_children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9)
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    if not keep_intermediate: shutil.rmtree(work, ignore_errors=True)
    log("done:", out, json.dumps(stats["timings_s"]), f"time to playback {stats['time_to_playback_s']}s")
    return stats


def add_words(new, pw, tot):
    """Word timestamps for newly final turns from the Parakeet words around them (align() on that window)."""
    import bisect
    starts = [w["start"] for w in pw]
    lo, hi = bisect.bisect_left(starts, new[0]["start"] - 15), bisect.bisect_right(starts, new[-1]["end"] + 15)
    turns, t = align(new, pw[lo:hi])
    for k in tot: tot[k] += t[k]
    return turns
