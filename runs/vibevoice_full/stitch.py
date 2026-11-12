# stitch chunks: offsets in seconds. Aligns speaker labels via overlap time-overlap, cuts at overlap midpoint.
import json, re, sys, collections
chunks = [("w/w50",0,3000),("w/cB",2940,2520),("w/cC",5400,2000)]
def segs(p, off):
    out=[]
    for s in json.load(open(p+".segs.json")):
        if s.get("speaker_id") is None: continue   # [Music] etc without speaker
        out.append(dict(start=round(s["start"]+off,2), end=round(s["end"]+off,2), speaker=int(s["speaker_id"]), text=s["text"].strip()))
    return out
def words_in(seglist, a, b):  # list of (speaker, mid time) for words in [a,b], time interpolated
    res=[]
    for s in seglist:
        w=s["text"].split()
        if not w: continue
        d=(s["end"]-s["start"])/len(w)
        for i,x in enumerate(w):
            t=s["start"]+d*(i+.5)
            if a<=t<=b: res.append((s["speaker"],t))
    return res
allc=[segs(p,o) for p,o,_ in chunks]
raw_flips=[]; fixed_flips=[]
mapping=[{0:0,1:1}]
for k in range(1,len(chunks)):
    a=chunks[k][1]; b=chunks[k-1][1]+chunks[k-1][2]   # overlap window
    prev=allc[k-1]; cur=allc[k]
    # time-overlap matrix between prev labels (already mapped) and cur labels
    M=collections.Counter()
    for s1 in prev:
        for s2 in cur:
            ov=min(s1["end"],s2["end"],b)-max(s1["start"],s2["start"],a)
            if ov>0: M[(s1["speaker"],s2["speaker"])]+=ov
    same=M[(0,0)]+M[(1,1)]; swap=M[(0,1)]+M[(1,0)]
    raw_flips.append(dict(overlap=(a,b),same=round(same,1),swap=round(swap,1)))
    flip = swap>same
    mp={0:1,1:0} if flip else {0:0,1:1}
    for s in cur: s["speaker"]=mp[s["speaker"]]
    mapping.append(mp)
    print(f"chunk{k} vs chunk{k-1} overlap {a}-{b}: agree {same:.1f}s disagree {swap:.1f}s -> flip={flip}")
# merge: keep chunk k segs whose start < cut, then chunk k+1 segs with start >= cut (cut = overlap midpoint)
final=[]
for k,c in enumerate(allc):
    hi = 1e9 if k==len(chunks)-1 else (chunks[k+1][1]+chunks[k][1]+chunks[k][2])/2
    last_end = final[-1]["end"] if final else -1
    for s in c:
        m=(s["start"]+s["end"])/2
        if s["start"]>=last_end-1.0 and (s["end"]<=hi or k==len(chunks)-1): final.append(s)
final.sort(key=lambda s:s["start"])
# ts fix: clamp overlaps
for a,b in zip(final,final[1:]):
    if b["start"]<a["end"]: a["end"]=b["start"]
def ts(x): return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{x%60:04.1f}"
json.dump([dict(start=s["start"],end=s["end"],speaker=f"SPEAKER_{s['speaker']:02d}",text=s["text"]) for s in final],open("transcript.json","w"),indent=1)
with open("transcript.txt","w") as f:
    for s in final: f.write(f"[{ts(s['start'])} - {ts(s['end'])}] SPEAKER_{s['speaker']:02d}: {s['text']}\n")
json.dump(dict(overlaps=raw_flips,mapping=mapping),open("stitch_stats.json","w"))
print(len(final),"segments",sum(len(s['text'].split()) for s in final),"words")
