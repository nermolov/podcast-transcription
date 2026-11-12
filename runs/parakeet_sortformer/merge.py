import sys, json, re, difflib, statistics
vv_path, words_path, out_prefix = sys.argv[1:4]
MARGIN = 1.0
vv = json.load(open(vv_path)); pw = json.load(open(words_path))
def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))
FILL = {"uh","um","uhm","mm","hmm","mhm","erm","ah"}
pn = [norm(w["w"]) for w in pw]
turns, stats = [], dict(vw=0, matched=0, interp=0)
cursor = 0
# global monotone alignment of all VV words against all Parakeet words, then window-filter per turn
gv = [(ti, wi, norm(x)) for ti, t in enumerate(vv) for wi, x in enumerate(t["text"].split())]
gk = [(i, k) for i, (_, _, k) in enumerate(gv) if k and k not in FILL]
pk = [(j, k) for j, k in enumerate(pn) if k and k not in FILL]
gsm = difflib.SequenceMatcher(None, [k for _, k in gk], [k for _, k in pk], autojunk=False)
GA = {}
for blk in gsm.get_matching_blocks():
    for q in range(blk.size):
        ti, wi, _ = gv[gk[blk.a + q][0]]
        j = pk[blk.b + q][0]
        t = vv[ti]
        if t["start"] - MARGIN <= pw[j]["start"] and pw[j]["end"] <= t["end"] + MARGIN:
            GA[(ti, wi)] = j
for ti, t in enumerate(vv):
    toks = t["text"].split()
    tn = [norm(x) for x in toks]
    lo = min([j for (x, _), j in GA.items() if x == ti], default=0)
    hi = max([j for (x, _), j in GA.items() if x == ti], default=-1) + 1
    anch = {wi: j for (x, wi), j in GA.items() if x == ti}
    ka = [x if x and x not in FILL else None for x in tn]
    # fuzzy rescue: unmatched word between anchors vs unmatched parakeet word in same gap (ratio>=0.6)
    keys = sorted(anch)
    bounds = [(-1, lo - 1 if anch else -1)] + [(i, anch[i]) for i in keys] + [(len(tn), hi if anch else 0)]
    for (ia, ja), (ib, jb) in zip(bounds, bounds[1:]):
        ai = [i for i in range(ia + 1, ib) if ka[i]]
        bj = [j for j in range(max(ja + 1, 0), min(jb, len(pw))) if pn[j] and pn[j] not in FILL and t["start"] - MARGIN <= pw[j]["start"] and pw[j]["end"] <= t["end"] + MARGIN]
        if len(ai) == len(bj) and ai:
            for i, j in zip(ai, bj):
                if difflib.SequenceMatcher(None, tn[i], pn[j]).ratio() >= 0.6: anch[i] = j
    # timestamps
    ws = [dict(w=toks[i], start=None, end=None, matched=False) for i in range(len(toks))]
    for i, j in anch.items():
        ws[i].update(start=pw[j]["start"], end=pw[j]["end"], matched=True)
    nm = len(anch)
    if nm:
        last = max(anch.values()); cursor = max(cursor, last + 1)
    # interpolate
    n = len(ws); i = 0
    while i < n:
        if ws[i]["start"] is not None: i += 1; continue
        j = i
        while j < n and ws[j]["start"] is None: j += 1
        left = ws[i-1]["end"] if i > 0 else None
        right = ws[j]["start"] if j < n else None
        if left is None: left = max(t["start"], (right - 0.3 * (j - i))) if right is not None else t["start"]
        if right is None: right = min(t["end"], left + 0.3 * (j - i)) if left is not None else t["end"]
        right = max(right, left)
        step = (right - left) / (j - i)
        for k in range(i, j):
            ws[k]["start"] = round(left + step * (k - i), 3); ws[k]["end"] = round(left + step * (k - i + 1), 3)
        i = j
    stats["vw"] += n; stats["matched"] += nm; stats["interp"] += n - nm
    turns.append(dict(turn=ti, speaker=t["speaker"], vv_start=t["start"], vv_end=t["end"], text=t["text"],
        aligned_start=ws[0]["start"] if ws else None, aligned_end=ws[-1]["end"] if ws else None,
        n_words=n, n_matched=nm, match_rate=round(nm / n, 3) if n else None, words=ws))
json.dump(turns, open(out_prefix + "merged.json", "w"), indent=1)
ts = lambda x: f"{int(x//60):02d}:{x%60:05.2f}"
with open(out_prefix + "merged.txt", "w") as f:
    for t in turns:
        f.write(f"[{ts(t['aligned_start'])} - {ts(t['aligned_end'])}] {t['speaker']}: {t['text']}\n")
# metrics
print(json.dumps(stats), "match%", round(100 * stats["matched"] / stats["vw"], 2), "pk_words", len(pw))
print("turns with match<0.8:", [(t["turn"], t["match_rate"], t["n_words"]) for t in turns if t["match_rate"] is not None and t["match_rate"] < 0.8])
ds = [t["aligned_start"] - t["vv_start"] for t in turns]; de = [t["aligned_end"] - t["vv_end"] for t in turns]
print("start off: mean %.2f med %.2f min %.2f max %.2f" % (statistics.mean(ds), statistics.median(ds), min(ds), max(ds)))
print("end off:   mean %.2f med %.2f min %.2f max %.2f" % (statistics.mean(de), statistics.median(de), min(de), max(de)))
print("turns |start off|>1s or |end off|>1s:", [(t["turn"], round(a, 2), round(b, 2), round(t["vv_end"]-t["vv_start"],1)) for t, a, b in zip(turns, ds, de) if abs(a) > 1 or abs(b) > 1])
# monotonic check
allw = [w for t in turns for w in t["words"]]
print("non-monotonic word starts:", sum(1 for a, b in zip(allw, allw[1:]) if b["start"] < a["start"] - 1e-6))
print("max interpolated run:", max((len(list(g)) for k, g in __import__('itertools').groupby(allw, key=lambda w: w["matched"]) if not k), default=0))
# subtitle split demo
def split(t, maxdur=6.0, maxw=14, gap=0.5):
    out, cur = [], []
    for w in t["words"]:
        if cur and (w["end"] - cur[0]["start"] > maxdur or len(cur) >= maxw or (w["start"] - cur[-1]["end"] > gap and len(cur) > 3) or cur[-1]["w"].endswith((".", "?", "!")) and len(cur) > 5):
            out.append(cur); cur = []
        cur.append(w)
    if cur: out.append(cur)
    return out
subs = [(t["speaker"], c) for t in turns for c in split(t)]
d = [c[-1]["end"] - c[0]["start"] for _, c in subs]
print("subtitle pieces", len(subs), "dur mean %.1f max %.1f" % (statistics.mean(d), max(d)))
with open(out_prefix + "subtitles.txt", "w") as f:
    for s, c in subs: f.write(f"[{ts(c[0]['start'])} - {ts(c[-1]['end'])}] {s}: {' '.join(w['w'] for w in c)}\n")
