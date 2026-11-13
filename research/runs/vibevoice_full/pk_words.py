import sys
from parakeet_mlx import from_pretrained
r = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3").transcribe(sys.argv[1], chunk_duration=120.0, overlap_duration=15.0)
import json; json.dump([dict(start=s.start,end=s.end,text=s.text) for s in r.sentences], open("parakeet_full_sentences.json","w"))
print("PK_WORDS", len(r.text.split()))
