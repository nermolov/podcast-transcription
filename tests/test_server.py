"""Server tests with a fake pipeline: `python -m pytest tests/test_server.py` (inside `nix develop`)."""
import hashlib, json, os, sqlite3, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from transcriber.pipeline import TranscribeError
from transcriber.server import create_app as _create_app

P = dict(chunk_minutes=50.0, overlap_seconds=60.0, context="", words=True)


def fake_run(input, outdir, *, progress=None, **kw):
    progress("prepare", None, "prep")
    if "FAIL" in input: raise TranscribeError("bad audio")
    if "BOOM" in input: raise RuntimeError("boom")
    progress("vibevoice", 0.5, "chunk 1/2")
    os.makedirs(outdir, exist_ok=True)
    json.dump({"turns": [], "kw": kw, "input": input}, open(os.path.join(outdir, "transcript.json"), "w"))
    open(os.path.join(outdir, "transcript.txt"), "w").write("hello\n")
    open(os.path.join(outdir, "subtitles.srt"), "w").write("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    json.dump({"n_turns": 0}, open(os.path.join(outdir, "stats.json"), "w"))
    return {}


def fake_fetch(url, work):
    if "FAIL" in url: raise TranscribeError("bad audio")
    if "BOOM" in url: raise RuntimeError("boom")
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "download.audio")
    open(path, "wb").write(url.encode())
    return path


def create_app(*a, **kw):
    return _create_app(*a, **{"fetch_fn": fake_fetch, **kw})


def sha(b): return hashlib.sha256(b).hexdigest()


def wait(c, id, want=("done", "failed"), h=None):
    for _ in range(250):
        j = c.get(f"/jobs/{id}", headers=h).json()
        if j["status"] in want: return j
        time.sleep(0.02)
    raise AssertionError(j)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(str(tmp_path), run_fn=fake_run)) as c:
        yield c


def test_upload_poll_results(client, tmp_path):
    r = client.post("/jobs", files={"file": ("ep.mp3", b"x" * 3_000_000, "audio/mpeg")},
                    data={"chunk_minutes": "20", "words": "false", "context": "Acme"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    id = r.json()["id"]
    j = wait(client, id)
    assert j["status"] == "done" and j["progress"] == 1.0 and j["finished_at"] and j["started_at"] and j["error"] is None
    assert j["params"] == dict(chunk_minutes=20.0, overlap_seconds=30.0, context="Acme", words=False, progressive=True, speakers="embed")
    r0 = client.get(f"/jobs/{id}/result").json()
    assert r0["kw"]["progressive"] is True and r0["kw"]["speakers"] == "embed"
    r = client.get(f"/jobs/{id}/result").json()
    assert r["input"].endswith(".mp3") and r["kw"]["words"] is False
    assert client.get(f"/jobs/{id}/result?format=txt").text == "hello\n"
    assert "-->" in client.get(f"/jobs/{id}/result?format=srt").text
    assert client.get(f"/jobs/{id}/result?format=stats").json() == {"n_turns": 0}
    assert client.get(f"/jobs/{id}/result?format=pdf").status_code == 400
    assert [x["id"] for x in client.get("/jobs").json()] == [id]
    assert os.listdir(tmp_path / "uploads") == []  # upload removed after success


def test_url_form_and_json(client):
    a = client.post("/jobs", data={"url": "http://example.com/a.mp3"}).json()["id"]
    b = client.post("/jobs", json={"url": "https://example.com/feed.xml", "overlap_seconds": 30}).json()["id"]
    assert wait(client, a)["status"] == "done"
    assert wait(client, b)["params"]["overlap_seconds"] == 30


def test_failures(client):
    a = client.post("/jobs", data={"url": "http://x/FAIL"}).json()["id"]
    b = client.post("/jobs", data={"url": "http://x/BOOM"}).json()["id"]
    j = wait(client, a); assert j["status"] == "failed" and j["error"] == "bad audio"
    assert wait(client, b)["error"] == "boom"
    assert client.get(f"/jobs/{a}/result").status_code == 409


@pytest.mark.parametrize("kw", [
    dict(data={}), dict(data={"url": "ftp://x"}), dict(data={"url": "http://x", "chunk_minutes": "abc"}),
    dict(data={"url": "http://x", "chunk_minutes": "10", "overlap_seconds": "700"}), dict(data={"url": "http://x", "words": "maybe"}),
    dict(files={"file": ("a.mp3", b"x")}, data={"url": "http://x"}), dict(json=[1]),
    dict(content=b"x", headers={"content-type": "text/plain"}),
])
def test_bad_request(client, kw):
    assert client.post("/jobs", **kw).status_code == 400
    assert client.get("/jobs").json() == []


def test_404_and_health(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.delete("/jobs/nope").status_code == 404
    assert client.get("/healthz").json()["ok"]


def test_auth(tmp_path):
    h = {"Authorization": "Bearer s3cret"}
    with TestClient(create_app(str(tmp_path), run_fn=fake_run, api_token="s3cret")) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/jobs").status_code == 401
        assert c.get("/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.post("/jobs", data={"url": "http://x/a"}).status_code == 401
        id = c.post("/jobs", data={"url": "http://x/a"}, headers=h).json()["id"]
        assert wait(c, id, h=h)["status"] == "done"
        assert c.get(f"/jobs/{id}/result", headers=h).status_code == 200
        assert c.delete(f"/jobs/{id}").status_code == 401
        assert c.delete(f"/jobs/{id}", headers=h).status_code == 200


def test_cancel_and_delete(tmp_path):
    gate = threading.Event()
    def slow(input, outdir, **kw):
        gate.wait(5); return fake_run(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=slow)) as c:
        a = c.post("/jobs", data={"url": "http://x/1"}).json()["id"]
        b = c.post("/jobs", files={"file": ("b.wav", b"abc")}).json()["id"]
        wait(c, a, want=("running",))
        assert c.get(f"/jobs/{b}").json()["status"] == "queued"
        assert c.delete(f"/jobs/{b}").status_code == 200  # queued -> cancelled
        assert c.get(f"/jobs/{b}").status_code == 404 and os.listdir(tmp_path / "uploads") == []
        assert c.delete(f"/jobs/{b}").status_code == 404
        gate.set()
        assert wait(c, a)["status"] == "done"
        tid = c.get(f"/jobs/{a}").json()["transcript_id"]
        assert os.path.isdir(tmp_path / "transcripts" / tid)
        assert c.delete(f"/jobs/{a}").json() == {"id": a, "deleted": True, "was": "done"}
        assert os.path.isdir(tmp_path / "transcripts" / tid) and c.get("/jobs").json() == []


def _child_run(pidfile):
    """run_fn that blocks in a pipeline child like the real VibeVoice worker, after writing to the job dir and chunk cache."""
    from transcriber.pipeline import run_child
    def run(input, outdir, *, progress=None, cache_dir=None, **kw):
        progress("vibevoice", 0.1, "chunk 1/1")
        os.makedirs(outdir, exist_ok=True); open(os.path.join(outdir, "partial.json"), "w").write("{}")
        open(os.path.join(cache_dir, "chunk0.json"), "w").write("{}")
        rc = run_child([sys.executable, "-c", f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(300)"])
        raise TranscribeError(f"worker failed (exit {rc})")
    return run


def _wait_pid(pidfile):
    for _ in range(250):
        if pidfile.exists() and pidfile.read_text(): return int(pidfile.read_text())
        time.sleep(0.02)
    raise AssertionError("child never started")


def test_cancel_running_kills_child_and_next_job_runs(tmp_path):
    pidfile = tmp_path / "child.pid"; slow = _child_run(pidfile)
    def run(input, outdir, **kw):
        return fake_run(input, outdir, **kw) if b"next" in open(input, "rb").read() else slow(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path / "s"), run_fn=run)) as c:
        a = up(c, b"long episode").json()["id"]
        pid = _wait_pid(pidfile); assert _alive(pid)
        b = c.post("/jobs", data={"url": "http://x/next"}).json()["id"]
        r = c.delete(f"/jobs/{a}")
        assert r.status_code == 200 and r.json() == {"id": a, "deleted": True, "was": "running"}
        assert not _alive(pid)  # DELETE returns only once the child is gone
        assert c.get(f"/jobs/{a}").status_code == 404
        assert wait(c, b)["status"] == "done"
        s = tmp_path / "s"
        assert not (s / "jobs" / a).exists() and os.listdir(s / "uploads") == []
        assert os.listdir(s / "cache" / "chunks" / sha(b"long episode")) == ["chunk0.json"]  # resumable later
        assert [j["id"] for j in c.get("/jobs").json()] == [b]
        # same audio again is a fresh job, not a dedupe hit on the cancelled one
        r = up(c, b"long episode")
        assert r.status_code == 202 and r.json()["id"] != a
        assert [x["id"] for x in c.delete("/jobs").json()["cancelled"]] == [r.json()["id"]]


def test_cancel_running_in_process_work(tmp_path):
    # no child to kill: the next progress() call stops the job, and nothing is published
    gate, entered = threading.Event(), threading.Event()
    def run(input, outdir, *, progress=None, **kw):
        progress("vibevoice", 0.1, "chunk 1/2"); entered.set(); gate.wait(5)
        progress("vibevoice", 0.5, "chunk 2/2")
        return fake_run(input, outdir, progress=progress, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        a = c.post("/jobs", data={"url": "http://x/a"}).json()["id"]
        assert entered.wait(5)
        assert c.delete(f"/jobs/{a}").json()["was"] == "running"
        gate.set()
        b = c.post("/jobs", data={"url": "http://x/b"}).json()["id"]
        assert wait(c, b)["status"] == "done"
        assert c.get(f"/jobs/{a}").status_code == 404 and len(os.listdir(tmp_path / "transcripts")) == 1
        assert not (tmp_path / "jobs" / a).exists()


def test_cancel_before_child_starts(tmp_path):
    # cancelled while still preparing (download, ffmpeg, model load): the child it was about to spawn must not start
    from transcriber.pipeline import run_child
    gate, entered, started = threading.Event(), threading.Event(), []
    def run(input, outdir, *, progress=None, **kw):
        if b"next" in open(input, "rb").read(): return fake_run(input, outdir, progress=progress, **kw)
        entered.set(); gate.wait(5)
        started.append(1); run_child([sys.executable, "-c", "import time; time.sleep(300)"])
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        a = c.post("/jobs", data={"url": "http://x/a"}).json()["id"]
        assert entered.wait(5)
        assert c.delete(f"/jobs/{a}").json()["was"] == "running"
        gate.set()
        b = c.post("/jobs", data={"url": "http://x/next"}).json()["id"]
        assert wait(c, b)["status"] == "done"  # and spawning works again for the next job
        assert started == [1] and c.get("/jobs").json()[0]["id"] == b


def test_cancel_all(tmp_path):
    pidfile = tmp_path / "child.pid"; slow = _child_run(pidfile)
    def run(input, outdir, **kw):
        return fake_run(input, outdir, **kw) if b"quick" in open(input, "rb").read() else slow(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path / "s"), run_fn=run)) as c:
        assert c.delete("/jobs").json() == {"cancelled": []}
        done = c.post("/jobs", data={"url": "http://x/quick1"}).json()["id"]
        assert wait(c, done)["status"] == "done"
        a = c.post("/jobs", data={"url": "http://x/long"}).json()["id"]
        pid = _wait_pid(pidfile)
        q1 = up(c, b"q1").json()["id"]; q2 = c.post("/jobs", data={"url": "http://x/q2"}).json()["id"]
        r = c.delete("/jobs")
        assert r.status_code == 200
        assert sorted((x["id"], x["was"]) for x in r.json()["cancelled"]) == sorted([(a, "running"), (q1, "queued"), (q2, "queued")])
        assert not _alive(pid)
        assert [j["id"] for j in c.get("/jobs").json()] == [done]  # finished jobs stay
        assert os.listdir(tmp_path / "s" / "uploads") == []
        fresh = c.post("/jobs", data={"url": "http://x/quick2"}).json()["id"]
        assert wait(c, fresh)["status"] == "done"
        h = c.get("/healthz").json(); assert h["queued"] == 0 and h["running"] == 0


def test_one_at_a_time(tmp_path):
    cur, peak, lock = [0], [0], threading.Lock()
    def f(input, outdir, **kw):
        with lock: cur[0] += 1; peak[0] = max(peak[0], cur[0])
        time.sleep(0.05)
        with lock: cur[0] -= 1
        return fake_run(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=f)) as c:
        ids = [c.post("/jobs", data={"url": f"http://x/{i}"}).json()["id"] for i in range(4)]
        assert all(wait(c, i)["status"] == "done" for i in ids)
    assert peak[0] == 1


def test_restart_requeues(tmp_path):
    # first "process" accepts jobs but never runs them; one is left `running` as if the process died mid-job
    s = create_app(str(tmp_path), run_fn=fake_run, start_worker=False).state.service
    s.submit("http://x/a", P, "aaa"); s.submit("http://x/b", P, "bbb")
    s.store.execute("UPDATE jobs SET status='running', stage='vibevoice', progress=0.4 WHERE id='aaa'")
    with TestClient(create_app(str(tmp_path), run_fn=fake_run)) as c:
        assert wait(c, "aaa")["status"] == "done" and wait(c, "bbb")["status"] == "done"
        assert len(c.get("/jobs").json()) == 2


def test_stats_input_is_original_name(tmp_path):
    seen = {}
    def run(input, outdir, *, progress=None, **kw): seen.update(kw, input=input); return {}
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        id = c.post("/jobs", files={"file": ("My Episode.mp3", b"x", "audio/mpeg")}).json()["id"]
        wait(c, id)
    assert seen["input_name"] == "My Episode.mp3" and seen["input"].endswith(id + ".mp3")
    assert seen["cache_dir"].endswith(os.path.join("cache", "chunks", sha(b"x")))


def _alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False


def test_shutdown_kills_child_and_requeues(tmp_path):
    from transcriber.pipeline import run_child
    pidfile = tmp_path / "child.pid"
    def slow(input, outdir, *, progress=None, **kw):
        progress("vibevoice", 0.1, "chunk 1/1")
        rc = run_child([sys.executable, "-c", f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(300)"])
        raise TranscribeError(f"worker failed (exit {rc})")  # what the real pipeline does when the child is killed
    with TestClient(create_app(str(tmp_path / "s"), run_fn=slow)) as c:
        id = c.post("/jobs", data={"url": "http://example.com/a.mp3"}).json()["id"]
        for _ in range(250):
            if pidfile.exists() and pidfile.read_text(): break
            time.sleep(0.02)
        pid = int(pidfile.read_text()); assert _alive(pid)
        svc = c.app.state.service
    # leaving the context runs lifespan shutdown
    for _ in range(100):
        if not _alive(pid): break
        time.sleep(0.05)
    assert not _alive(pid)
    row = svc.store.get(id)
    assert row["status"] == "queued" and row["error"] is None
    # next start picks it up again
    done = []
    with TestClient(create_app(str(tmp_path / "s"), run_fn=lambda *a, **k: done.append(1) or {})) as c2:
        assert wait(c2, id)["status"] == "done"


def test_atexit_kills_child_when_parent_dies(tmp_path):
    import signal, subprocess
    pidfile = tmp_path / "pid"
    code = ("import sys,os,time,threading\nsys.path.insert(0,%r)\nfrom transcriber.pipeline import run_child, install_signal_handlers\ninstall_signal_handlers()\n"
            "threading.Thread(target=lambda: run_child([sys.executable,'-c',\"import os,time;open(%r,'w').write(str(os.getpid()));time.sleep(300)\"]),daemon=True).start()\n"
            "time.sleep(300)\n") % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), str(pidfile))
    p = subprocess.Popen([sys.executable, "-c", code])
    for _ in range(250):
        if pidfile.exists() and pidfile.read_text(): break
        time.sleep(0.04)
    pid = int(pidfile.read_text()); assert _alive(pid)
    p.send_signal(signal.SIGTERM); p.wait(10)  
    for _ in range(100):
        if not _alive(pid): break
        time.sleep(0.05)
    assert not _alive(pid)


def up(c, data=b"audio-1", **kw):
    f = kw.pop("data_fields", {})
    return c.post("/jobs", files={"file": ("ep.mp3", data, "audio/mpeg")}, data=f, **kw)


def test_duplicate_upload_returns_cached_done(client, tmp_path):
    runs = []
    client.app.state.service.run_fn = lambda *a, **k: runs.append(1) or fake_run(*a, **k)
    first = up(client, data_fields={"key": "k1"}).json(); j = wait(client, first["id"])
    r = up(client, data_fields={"key": "k2"})
    b = r.json()
    assert r.status_code == 200 and b["cached"] is True and b["status"] == "done" and b["id"] != first["id"] and b["transcript_id"] == j["transcript_id"]
    j2 = client.get(f"/jobs/{b['id']}").json()
    assert j2["cached"] is True and j2["overall_progress"] == 1.0 and j2["sha256"] == sha(b"audio-1") and j2["key"] == "k2" and j2["audio_s"] is None
    assert client.get(f"/jobs/{b['id']}/result?format=txt").text == "hello\n"
    assert len(runs) == 1 and os.listdir(tmp_path / "uploads") == []
    assert client.get("/transcripts", params={"key": "k2"}).json()["transcript_id"] == j["transcript_id"]


def test_duplicate_while_queued_returns_active_job(tmp_path):
    gate = threading.Event()
    def slow(input, outdir, **kw):
        gate.wait(5); return fake_run(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=slow)) as c:
        a = up(c, b"first").json()
        wait(c, a["id"], want=("running",))
        b = up(c, b"second").json()
        assert client_status(c, b["id"]) == "queued" and c.get(f"/jobs/{b['id']}").json()["queue_position"] == 1
        assert c.get(f"/jobs/{a['id']}").json()["queue_position"] is None
        r = up(c, b"second"); assert r.status_code == 200 and r.json() == {"id": b["id"], "status": "queued", "cached": True}
        r = up(c, b"first"); assert r.json()["id"] == a["id"] and r.json()["status"] == "running"
        assert len(c.get("/jobs").json()) == 2 and len(os.listdir(tmp_path / "uploads")) == 2
        assert c.get("/transcripts", params={"sha256": sha(b"second")}).json()["status"] == "queued"
        gate.set(); wait(c, a["id"]); wait(c, b["id"])


def client_status(c, id): return c.get(f"/jobs/{id}").json()["status"]


def test_sha256_mismatch_and_match(client, tmp_path):
    r = up(client, data_fields={"sha256": "0" * 64})
    assert r.status_code == 400 and os.listdir(tmp_path / "uploads") == [] and client.get("/jobs").json() == []
    assert up(client, data_fields={"sha256": "xyz"}).status_code == 400
    r = up(client, data_fields={"sha256": sha(b"audio-1")})
    assert r.status_code == 202 and r.json()["cached"] is False


def test_force_bypasses_dedupe(client):
    a = up(client).json(); ta = wait(client, a["id"])["transcript_id"]
    b = up(client, data_fields={"force": "true"})
    assert b.status_code == 202 and b.json()["id"] != a["id"]
    tb = wait(client, b.json()["id"])
    assert tb["cached"] is False and tb["transcript_id"] != ta
    assert client.get(f"/transcripts/{ta}/result").status_code == 404  # superseded
    assert client.get("/transcripts", params={"sha256": sha(b"audio-1")}).json()["transcript_id"] == tb["transcript_id"]


def test_words_true_result_satisfies_words_false(client):
    a = up(client).json(); wait(client, a["id"])
    r = up(client, data_fields={"words": "false"})
    assert r.status_code == 200 and r.json()["cached"] is True
    r = up(client, data_fields={"chunk_minutes": "20"})
    assert r.status_code == 202


def test_lookup_by_sha256_and_key(client):
    assert client.get("/transcripts", params={"sha256": sha(b"audio-1")}).status_code == 404
    assert client.get("/transcripts", params={"key": "kk"}).status_code == 404
    assert client.get("/transcripts").status_code == 400
    id = up(client, data_fields={"key": "kk"}).json()["id"]
    j = wait(client, id)
    for params in ({"sha256": sha(b"audio-1")}, {"key": "kk"}, {"sha256": sha(b"audio-1"), "key": "other"}):
        r = client.get("/transcripts", params=params)
        assert r.status_code == 200
        b = r.json(); assert b["status"] == "done" and b["transcript_id"] == j["transcript_id"] and b["job_id"] == id and b["sha256"] == sha(b"audio-1")
    assert client.get("/transcripts", params={"key": "kk"}).json()["key"] == "kk"
    assert client.get("/transcripts", params={"sha256": sha(b"audio-2"), "key": "kk"}).status_code == 404  # sha takes priority
    assert client.get(f"/transcripts/{j['transcript_id']}/result?format=srt").status_code == 200
    assert client.get(f"/transcripts/{j['transcript_id']}/result?format=pdf").status_code == 400
    assert client.get("/transcripts/nope/result").status_code == 404


def test_delete_job_keeps_transcript_and_delete_transcript(client, tmp_path):
    id = up(client).json()["id"]; tid = wait(client, id)["transcript_id"]
    assert client.delete(f"/jobs/{id}").status_code == 200
    assert client.get(f"/jobs/{id}/result").status_code == 404
    assert client.get("/transcripts", params={"sha256": sha(b"audio-1")}).json()["job_id"] is None
    assert client.get(f"/transcripts/{tid}/result").json()["kw"]["words"] is True
    assert client.delete(f"/transcripts/{tid}").status_code == 200
    assert not os.path.exists(tmp_path / "transcripts" / tid) and client.delete(f"/transcripts/{tid}").status_code == 404
    assert client.get("/transcripts", params={"sha256": sha(b"audio-1")}).status_code == 404


def test_delete_transcript_conflicts_with_active_job(tmp_path):
    gate = threading.Event()
    def slow(input, outdir, **kw):
        if gate.is_set() is False and "again" in open(input, "rb").read().decode(): gate.wait(5)
        return fake_run(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=slow)) as c:
        tid = wait(c, up(c, b"again").json()["id"], want=("running", "done"))
        gate.set(); tid = wait(c, tid["id"])["transcript_id"]
        gate.clear()
        b = up(c, b"again", data_fields={"force": "1"}).json(); wait(c, b["id"], want=("running",))
        assert c.delete(f"/transcripts/{tid}").status_code == 409
        gate.set(); wait(c, b["id"])


def test_url_dedupe_by_key_url_and_content(client):
    runs = []
    client.app.state.service.run_fn = lambda *a, **k: runs.append(1) or fake_run(*a, **k)
    a = client.post("/jobs", data={"url": "http://Example.com:80/a.mp3#frag", "key": "ek"}).json()
    j = wait(client, a["id"])
    r = client.post("/jobs", data={"url": "http://example.com/a.mp3"}); assert r.status_code == 200 and r.json()["transcript_id"] == j["transcript_id"]
    r = client.post("/jobs", data={"url": "http://other.example/z.mp3", "key": "ek"}); assert r.status_code == 200 and r.json()["cached"]
    assert client.post("/jobs", data={"url": "http://example.com/a.mp3", "force": "true"}).status_code == 202
    assert client.get("/transcripts", params={"key": "ek"}).json()["transcript_id"]
    assert client.get("/transcripts", params={"key": "http://example.com/a.mp3"}).status_code == 404
    # same bytes behind a different URL: content-hash dedupe after download, no model run
    client.app.state.service.fetch_fn = lambda url, work: fake_fetch("http://example.com/a.mp3", work)
    n = len(runs); c2 = client.post("/jobs", data={"url": "http://elsewhere.test/copy.mp3"}).json()
    j2 = wait(client, c2["id"]); assert j2["cached"] is True and j2["sha256"] == sha(b"http://example.com/a.mp3") and j2["overall_progress"] == 1.0
    assert len(runs) == n


def test_chunk_cache_dir_keyed_by_hash_and_shared(tmp_path):
    seen = []
    def run(input, outdir, *, cache_dir=None, **kw):
        seen.append(cache_dir); os.makedirs(cache_dir, exist_ok=True); return fake_run(input, outdir, cache_dir=cache_dir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        wait(c, up(c, b"zz").json()["id"]); wait(c, up(c, b"zz", data_fields={"words": "false", "chunk_minutes": "30"}).json()["id"])
    d = str(tmp_path / "cache" / "chunks" / sha(b"zz"))
    assert seen == [d, d] and os.path.isdir(d)


def test_eviction_lru_by_size(tmp_path):
    with TestClient(create_app(str(tmp_path), run_fn=fake_run)) as c:
        svc = c.app.state.service
        a = wait(c, up(c, b"one").json()["id"])["transcript_id"]
        b = wait(c, up(c, b"two").json()["id"])["transcript_id"]
        size = svc.store.transcript(a)["size_bytes"]; assert size > 0
        c.get(f"/transcripts/{a}/result")  # a becomes the most recently used
        svc.max_bytes = size + 1
        assert svc.evict() >= 1
        assert c.get(f"/transcripts/{b}/result").status_code == 404 and c.get(f"/transcripts/{a}/result").status_code == 200
        assert not os.path.exists(tmp_path / "cache" / "chunks" / sha(b"two"))
        assert c.get("/transcripts", params={"sha256": sha(b"two")}).status_code == 404


def test_eviction_ttl_and_keep_newest(tmp_path):
    with TestClient(create_app(str(tmp_path), run_fn=fake_run, cache_max_gb=1e-9)) as c:
        id = up(c, b"only").json()["id"]
        assert wait(c, id)["status"] == "done"
        time.sleep(0.2)
        assert c.get("/transcripts", params={"sha256": sha(b"only")}).status_code == 200  # just-finished one survives the post-job eviction
        svc = c.app.state.service
        svc.ttl_days = 1e-9; svc.max_bytes = 1e12
        assert svc.evict() >= 1 and c.get("/transcripts", params={"sha256": sha(b"only")}).status_code == 404


def test_migration_from_old_schema(tmp_path):
    (tmp_path / "jobs" / "olddone").mkdir(parents=True)
    for n, t in (("transcript.json", "[]"), ("transcript.txt", "hi\n"), ("subtitles.srt", "1\n"), ("stats.json", '{"audio_s": 12.5}')):
        (tmp_path / "jobs" / "olddone" / n).write_text(t)
    db = sqlite3.connect(tmp_path / "jobs.db")
    db.execute("""CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT, progress REAL,
        message TEXT, error TEXT, source TEXT NOT NULL, params TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL)""")
    p = json.dumps(P)
    for id, st, src in (("olddone", "done", "http://example.com/old.mp3"), ("goneone", "done", "/gone/up.mp3"), ("oldfail", "failed", "http://x/f")):
        db.execute("INSERT INTO jobs (id,status,stage,progress,source,params,created_at,finished_at) VALUES (?,?,?,?,?,?,?,?)", (id, st, st, 1.0, src, p, 1000.0, 1001.0))
    db.commit(); db.close()
    with TestClient(create_app(str(tmp_path), run_fn=fake_run)) as c:
        j = c.get("/jobs/olddone").json()
        assert j["status"] == "done" and j["transcript_id"] == "olddone" and j["audio_s"] == 12.5 and j["sha256"] is None
        assert c.get("/jobs/olddone/result?format=txt").text == "hi\n"
        assert c.get("/transcripts/olddone/result?format=stats").json() == {"audio_s": 12.5}
        assert c.get("/jobs/goneone").json()["transcript_id"] is None and c.get("/jobs/goneone/result").status_code == 404
        assert c.get("/jobs/oldfail").json()["status"] == "failed"
        assert c.post("/jobs", data={"url": "http://example.com/old.mp3"}).status_code == 202  # new default settings: not the old result
        r = c.post("/jobs", data={"url": "http://example.com/old.mp3", "chunk_minutes": "50", "overlap_seconds": "60", "progressive": "false", "speakers": "overlap"})
        assert r.status_code == 200 and r.json()["transcript_id"] == "olddone"
        assert not os.path.exists(tmp_path / "jobs" / "olddone")


def test_bad_speakers_param(client):
    assert up(client, data_fields={"speakers": "magic"}).status_code == 400
    assert up(client, data_fields={"progressive": "maybe"}).status_code == 400
    assert up(client, data_fields={"speakers": "overlap", "progressive": "false"}).status_code == 202


def test_partial_transcript_while_running(tmp_path):
    gate, half = threading.Event(), threading.Event()
    turns = [dict(speaker="SPEAKER_00", start=0.0, end=2.0, text="hello"), dict(speaker="SPEAKER_01", start=2.0, end=5.0, text="hi there")]
    def run(input, outdir, *, progress=None, **kw):
        progress("prepare", None, "prep")
        os.makedirs(outdir, exist_ok=True)
        json.dump(dict(complete=False, audio_s=60.0, ready_until=2.0, speakers=["SPEAKER_00"], turns=turns[:1]), open(os.path.join(outdir, "partial.json"), "w"))
        progress("vibevoice", 2.0 / 60, "final up to 2s", ready_s=2.0, audio_s=60.0)
        half.set(); gate.wait(5)
        json.dump(turns, open(os.path.join(outdir, "transcript.json"), "w"))
        for n in ("transcript.txt", "subtitles.srt"): open(os.path.join(outdir, n), "w").write("x\n")
        json.dump({"audio_s": 60.0}, open(os.path.join(outdir, "stats.json"), "w"))
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        id = up(c).json()["id"]
        assert half.wait(5)
        j = c.get(f"/jobs/{id}").json()
        assert j["status"] == "running" and j["ready_s"] == 2.0 and j["audio_s"] == 60.0 and 0.02 < j["overall_progress"] < 0.1
        p = c.get(f"/jobs/{id}/partial").json()
        assert p["status"] == "running" and p["complete"] is False and p["ready_until"] == 2.0 and [t["text"] for t in p["turns"]] == ["hello"]
        gate.set(); wait(c, id)
        p = c.get(f"/jobs/{id}/partial", params={"since": 1}).json()
        assert p["complete"] is True and p["ready_until"] == 60.0 and p["n_turns"] == 2 and [t["text"] for t in p["turns"]] == ["hi there"]
        assert p["speakers"] == ["SPEAKER_00", "SPEAKER_01"] and c.get(f"/jobs/{id}").json()["ready_s"] == 60.0


def test_partial_queued_and_failed(tmp_path):
    gate = threading.Event()
    def run(input, outdir, **kw):
        if "FAIL" in open(input, "rb").read().decode(): raise TranscribeError("bad audio")
        gate.wait(5); return fake_run(input, outdir, **kw)
    with TestClient(create_app(str(tmp_path), run_fn=run)) as c:
        a = up(c, b"first").json()["id"]; wait(c, a, want=("running",))
        b = up(c, b"second").json()["id"]
        p = c.get(f"/jobs/{b}/partial").json()
        assert p["status"] == "queued" and p["turns"] == [] and p["ready_until"] == 0.0
        assert c.get(f"/jobs/{a}/partial").json()["turns"] == []  # running, nothing final yet
        gate.set(); wait(c, a); wait(c, b)
        f = up(c, b"FAIL").json()["id"]; wait(c, f)
        assert c.get(f"/jobs/{f}/partial").status_code == 409 and c.get("/jobs/nope/partial").status_code == 404
