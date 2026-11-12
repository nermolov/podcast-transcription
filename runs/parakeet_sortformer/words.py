import sys, json
from parakeet_mlx import from_pretrained
asr = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
res = asr.transcribe(sys.argv[1], chunk_duration=120.0, overlap_duration=15.0)
words = []
for s in res.sentences:
    cur = None
    for tok in s.tokens:
        if cur is None or tok.text.startswith(" "):
            if cur: words.append(cur)
            cur = dict(w=tok.text.strip(), start=round(tok.start,3), end=round(tok.end,3))
        else:
            cur["w"] += tok.text; cur["end"] = round(tok.end,3)
    if cur: words.append(cur)
json.dump([w for w in words if w["w"]], open(sys.argv[2], "w"))
print(len(words))
