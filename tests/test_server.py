"""Server tests with a fake pipeline: `python -m pytest tests/test_server.py` (inside `nix develop`)."""
import json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from fastapi.testclient import TestClient
from transcriber.pipeline import TranscribeError
from transcriber.server import create_app

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
    assert j["params"] == dict(chunk_minutes=20.0, overlap_seconds=60.0, context="Acme", words=False)
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
        assert c.delete(f"/jobs/{a}").status_code == 409  # running
        assert c.delete(f"/jobs/{b}").status_code == 200  # queued -> cancelled
        assert c.get(f"/jobs/{b}").status_code == 404 and os.listdir(tmp_path / "uploads") == []
        gate.set()
        assert wait(c, a)["status"] == "done"
        assert os.path.isdir(tmp_path / "jobs" / a)
        assert c.delete(f"/jobs/{a}").status_code == 200
        assert not os.path.exists(tmp_path / "jobs" / a) and c.get("/jobs").json() == []


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
