"""Measure time to playback from a client's point of view against the running HTTP service.

    python research/streaming/client_probe.py AUDIO --url http://127.0.0.1:8000 [--token T] [--out probe.json] [form fields...]

Uploads AUDIO (POST /jobs), then polls GET /jobs/{id}/partial?since=n every --interval seconds and records when each turn was
first seen (seconds since the upload finished). Time to playback = max over turns of (first seen - turn start): the earliest
moment after the upload a player could start without ever reaching a turn that is not there yet (at this polling interval).
Extra form fields as key=value (e.g. speakers=overlap).
"""
import argparse, json, time, urllib.request, uuid


def call(url, token, method="GET", body=None, ctype=None):
    req = urllib.request.Request(url, data=body, method=method)
    if token: req.add_header("Authorization", f"Bearer {token}")
    if ctype: req.add_header("Content-Type", ctype)
    with urllib.request.urlopen(req) as r: return json.loads(r.read())


def multipart(path, fields):
    b = uuid.uuid4().hex; parts = []
    for k, v in fields.items(): parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{path.split("/")[-1]}"\r\nContent-Type: audio/mpeg\r\n\r\n'.encode()
                 + open(path, "rb").read() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("audio"); ap.add_argument("fields", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--token"); ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    body, ctype = multipart(a.audio, dict(f.split("=", 1) for f in a.fields))
    j = call(f"{a.url}/jobs", a.token, "POST", body, ctype); t0 = time.time()
    print("job", j, flush=True)
    seen, timeline, n = [], [], 0
    while True:
        p = call(f"{a.url}/jobs/{j['id']}/partial?since={n}", a.token); t = time.time() - t0
        for x in p["turns"]: seen.append(dict(start=x["start"], end=x["end"], speaker=x["speaker"], seen=round(t, 2)))
        n += len(p["turns"]); timeline.append((round(t, 2), p["ready_until"], n, p["status"]))
        if n and len(timeline) % 30 == 1: print(f"t={t:6.1f}s ready_until={p['ready_until']:7.1f}s turns={n} {p['status']}", flush=True)
        if p["complete"] or p["status"] == "failed": break
        time.sleep(a.interval)
    ttp = max(x["seen"] - x["start"] for x in seen)
    worst = max(seen, key=lambda x: x["seen"] - x["start"])
    res = dict(job=j["id"], time_to_playback_s=round(ttp, 1), first_turn_seen_s=seen[0]["seen"], worst_turn=worst, total_s=timeline[-1][0],
               n_turns=n, speakers=sorted({x["speaker"] for x in seen}), timeline=timeline)
    print(json.dumps({k: v for k, v in res.items() if k != "timeline"}, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
