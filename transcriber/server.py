"""HTTP job service over transcriber.run: one background worker, SQLite-persisted jobs.

Env: STATE_DIR (default ~/.local/state/transcribe-service), HOST (127.0.0.1), PORT (8000), API_TOKEN (optional bearer auth).
Layout: STATE_DIR/uploads/<id><ext>, STATE_DIR/jobs/<id>/ (pipeline outdir), STATE_DIR/jobs.db.
"""
import json, os, queue, re, secrets, shutil, sqlite3, threading, time, traceback, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

FORMATS = {"json": ("transcript.json", "application/json"), "txt": ("transcript.txt", "text/plain; charset=utf-8"),
           "srt": ("subtitles.srt", "application/x-subrip"), "stats": ("stats.json", "application/json")}
COLS = "id,status,stage,progress,message,error,source,params,created_at,started_at,finished_at"


def _iso(t):
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thread-safe SQLite job table."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT, progress REAL,
            message TEXT, error TEXT, source TEXT NOT NULL, params TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL)""")

    def q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).rowcount

    def get(self, id):
        r = self.q(f"SELECT {COLS} FROM jobs WHERE id=?", (id,))
        return r[0] if r else None

    def insert(self, id, source, params):
        self.execute("INSERT INTO jobs (id,status,stage,source,params,created_at) VALUES (?,?,?,?,?,?)",
                     (id, "queued", "queued", source, json.dumps(params), time.time()))


def public(r):
    return dict(id=r["id"], status=r["status"], stage=r["stage"], progress=r["progress"], message=r["message"], error=r["error"],
                params=json.loads(r["params"]), created_at=_iso(r["created_at"]), started_at=_iso(r["started_at"]),
                finished_at=_iso(r["finished_at"]))


def default_run(*a, **kw):
    from transcriber.pipeline import run  # lazy: keeps app import light
    return run(*a, **kw)


class Service:
    def __init__(self, state_dir, run_fn=None, start_worker=True):
        self.state = os.path.abspath(state_dir)
        self.uploads = os.path.join(self.state, "uploads"); self.jobs = os.path.join(self.state, "jobs")
        for d in (self.uploads, self.jobs): os.makedirs(d, exist_ok=True)
        self.store = Store(os.path.join(self.state, "jobs.db"))
        self.run_fn = run_fn or default_run
        self.queue = queue.Queue(); self.stop = threading.Event(); self.thread = None
        # crash/restart recovery: running jobs resume (chunk cache makes that cheap), queued jobs keep their order
        self.store.execute("UPDATE jobs SET status='queued', stage='queued', message='re-queued after restart', started_at=NULL WHERE status='running'")
        for r in self.store.q("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"): self.queue.put(r["id"])
        if start_worker: self.start()

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="transcribe-worker", daemon=True); self.thread.start()

    def shutdown(self):
        self.stop.set(); self.queue.put(None)

    def submit(self, source, params, id=None):
        id = id or uuid.uuid4().hex[:12]
        self.store.insert(id, source, params); self.queue.put(id)
        return id

    def job_dir(self, id): return os.path.join(self.jobs, id)

    def remove_files(self, id):
        shutil.rmtree(self.job_dir(id), ignore_errors=True)
        for f in os.listdir(self.uploads):
            if f.split(".")[0] == id: os.remove(os.path.join(self.uploads, f))

    def _loop(self):
        while not self.stop.is_set():
            id = self.queue.get()
            if id is None: break
            # claim atomically; skips jobs cancelled/deleted while queued
            if self.store.execute("UPDATE jobs SET status='running', stage='starting', started_at=?, finished_at=NULL WHERE id=? AND status='queued'", (time.time(), id)) != 1:
                continue
            self._run(id)

    def _run(self, id):
        r = self.store.get(id); p = json.loads(r["params"]); outdir = self.job_dir(id)
        def progress(stage, fraction, message):
            self.store.execute("UPDATE jobs SET stage=?, progress=?, message=? WHERE id=? AND status='running'", (stage, fraction, message, id))
        try:
            self.run_fn(r["source"], outdir, chunk_minutes=p["chunk_minutes"], overlap_seconds=p["overlap_seconds"], context=p["context"],
                        words=p["words"], progress=progress)
            self.store.execute("UPDATE jobs SET status='done', stage='done', progress=1.0, message='finished', finished_at=? WHERE id=? AND status='running'", (time.time(), id))
            if not r["source"].startswith(("http://", "https://")):
                try: os.remove(r["source"])  # upload no longer needed once done
                except OSError: pass
        except BaseException as e:
            traceback.print_exc()
            msg = str(e) or type(e).__name__
            self.store.execute("UPDATE jobs SET status='failed', error=?, message=?, finished_at=? WHERE id=? AND status='running'", (msg, msg, time.time(), id))


def _num(v, name, default, lo, hi, hi_incl=True):
    if v in (None, ""): return default
    try: x = float(v)
    except (TypeError, ValueError): raise HTTPException(400, f"{name} must be a number")
    if not (lo < x <= hi if hi_incl else lo < x < hi): raise HTTPException(400, f"{name} out of range")
    return x


def _bool(v, default):
    if v in (None, ""): return default
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"): return True
    if s in ("0", "false", "no", "off"): return False
    raise HTTPException(400, "words must be a boolean")


def create_app(state_dir=None, run_fn=None, api_token=None, start_worker=True):
    state_dir = state_dir or os.environ.get("STATE_DIR") or os.path.expanduser("~/.local/state/transcribe-service")
    api_token = api_token if api_token is not None else (os.environ.get("API_TOKEN") or None)
    svc = Service(state_dir, run_fn, start_worker)

    @asynccontextmanager
    async def lifespan(app):
        yield
        svc.shutdown()

    def auth(request: Request):
        if not api_token: return
        h = request.headers.get("authorization", "")
        if not (h.lower().startswith("bearer ") and secrets.compare_digest(h[7:].strip(), api_token)):
            raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="transcribe service", lifespan=lifespan)
    app.state.service = svc

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "queued": len(svc.store.q("SELECT 1 FROM jobs WHERE status='queued'")),
                "running": len(svc.store.q("SELECT 1 FROM jobs WHERE status='running'"))}

    @app.post("/jobs", status_code=202, dependencies=[Depends(auth)])
    async def create_job(request: Request):
        ctype = request.headers.get("content-type", "")
        upload = None
        if ctype.startswith("application/json"):
            try: data = await request.json()
            except ValueError: raise HTTPException(400, "invalid JSON")
            if not isinstance(data, dict): raise HTTPException(400, "JSON body must be an object")
        elif ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            form = await request.form()  # file parts are spooled to disk by Starlette, not held in memory
            data = dict(form); upload = data.pop("file", None)
            if upload is not None and not isinstance(upload, UploadFile): raise HTTPException(400, "`file` must be a file upload")
        else:
            raise HTTPException(400, "send multipart/form-data (file or url), form fields, or JSON with `url`")
        url = data.get("url") or None
        if upload is None and not url: raise HTTPException(400, "provide either a `file` upload or a `url`")
        if upload is not None and url: raise HTTPException(400, "provide only one of `file` and `url`")
        if url and not re.match(r"https?://", str(url)): raise HTTPException(400, "`url` must be http(s)")
        chunk = _num(data.get("chunk_minutes"), "chunk_minutes", 50.0, 0, 59)
        ov = _num(data.get("overlap_seconds"), "overlap_seconds", 60.0, -1, chunk * 60, hi_incl=False)
        if ov < 0: raise HTTPException(400, "overlap_seconds out of range")
        params = dict(chunk_minutes=chunk, overlap_seconds=ov, context=str(data.get("context") or ""), words=_bool(data.get("words"), True))
        id = uuid.uuid4().hex[:12]
        if upload is not None:
            ext = re.sub(r"[^A-Za-z0-9.]", "", os.path.splitext(upload.filename or "")[1])[:8]
            source = os.path.join(svc.uploads, id + ext)
            def save():
                with open(source, "wb") as f: shutil.copyfileobj(upload.file, f, 1 << 20)
            await run_in_threadpool(save)
            await upload.close()
        else:
            source = str(url)
        svc.submit(source, params, id)
        return {"id": id, "status": "queued"}

    def need(id):
        r = svc.store.get(id)
        if r is None: raise HTTPException(404, "no such job")
        return r

    @app.get("/jobs", dependencies=[Depends(auth)])
    def list_jobs():
        return [public(r) for r in svc.store.q(f"SELECT {COLS} FROM jobs ORDER BY created_at DESC")]

    @app.get("/jobs/{id}", dependencies=[Depends(auth)])
    def get_job(id: str):
        return public(need(id))

    @app.get("/jobs/{id}/result", dependencies=[Depends(auth)])
    def result(id: str, format: str = "json"):
        r = need(id)
        if format not in FORMATS: raise HTTPException(400, f"format must be one of {sorted(FORMATS)}")
        if r["status"] != "done": raise HTTPException(409, f"job is {r['status']}, not done")
        name, mt = FORMATS[format]; path = os.path.join(svc.job_dir(id), name)
        if not os.path.exists(path): raise HTTPException(404, f"{name} missing")
        return FileResponse(path, media_type=mt)

    @app.delete("/jobs/{id}", dependencies=[Depends(auth)])
    def delete(id: str):
        r = need(id)
        if r["status"] == "running": raise HTTPException(409, "job is running and cannot be cancelled")
        svc.store.execute("DELETE FROM jobs WHERE id=? AND status!='running'", (id,))
        svc.remove_files(id)
        return {"id": id, "deleted": True, "was": r["status"]}

    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
