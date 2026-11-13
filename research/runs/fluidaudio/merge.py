import json,sys
asr=json.load(open(sys.argv[1])); dia=json.load(open(sys.argv[2])); out=sys.argv[3]
segs=sorted(dia['segments'],key=lambda s:s['startTimeSeconds'])
names={}
def spk(w):
    best,bo=None,0
    for s in segs:
        o=min(w['endTime'],s['endTimeSeconds'])-max(w['startTime'],s['startTimeSeconds'])
        if o>bo: best,bo=s['speakerId'],o
    if best is None:  # nearest segment
        m=(w['startTime']+w['endTime'])/2
        best=min(segs,key=lambda s:min(abs(m-s['startTimeSeconds']),abs(m-s['endTimeSeconds'])))['speakerId']
    return best
groups=[]
for w in asr['wordTimings']:
    s=spk(w); s=names.setdefault(s,'SPEAKER_%02d'%len(names))
    if groups and groups[-1]['speaker']==s and w['startTime']-groups[-1]['end']<2.0:
        g=groups[-1]; g['end']=w['endTime']; g['text']+=' '+w['word']
    else: groups.append(dict(start=w['startTime'],end=w['endTime'],speaker=s,text=w['word']))
json.dump(groups,open(out+'.json','w'),indent=1)
f=lambda t:'%02d:%04.1f'%(t//60,t%60)
open(out+'.txt','w').write(''.join(f"[{f(g['start'])} - {f(g['end'])}] {g['speaker']}: {g['text']}\n" for g in groups))
