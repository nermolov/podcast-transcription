"""HTTP job service over transcriber.run: one background worker, SQLite-persisted jobs and a deduplicated transcript cache.

Env: STATE_DIR (default ~/.local/state/transcribe-service), HOST (127.0.0.1), PORT (8000), API_TOKEN (optional bearer auth),
CACHE_MAX_GB (20; LRU limit for transcripts + chunk cache), CACHE_TTL_DAYS (optional; evict entries unused for that long).
Layout: STATE_DIR/uploads/<id><ext>, STATE_DIR/jobs/<id>/ (pipeline outdir while running), STATE_DIR/transcripts/<tid>/ (results),
STATE_DIR/cache/chunks/<audio sha256>/ (VibeVoice chunk cache shared across jobs), STATE_DIR/jobs.db.
"""
import atexit, hashlib, json, os, queue, re, secrets, shutil, sqlite3, threading, time, traceback, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile

from transcriber.pipeline import MODEL_DEFAULT, PIPELINE_VERSION

FORMATS = {"json": ("transcript.json", "application/json"), "txt": ("transcript.txt", "text/plain; charset=utf-8"),
           "srt": ("subtitles.srt", "application/x-subrip"), "stats": ("stats.json", "application/json")}
COLS = "id,status,stage,progress,message,error,source,params,created_at,started_at,finished_at,sha256,ekey,transcript_id,cached,audio_s,name,forced"
ACTIVE = "status IN ('queued','running')"
DEFAULT_PARAMS = dict(chunk_minutes=50.0, overlap_seconds=60.0, context="", words=True)
STAGES = {"prepare": (0.0, 0.03), "vibevoice": (0.03, 0.80), "parakeet": (0.80, 0.95), "align": (0.95, 0.99), "write": (0.99, 1.0)}
NEW_COLS = dict(name="TEXT", sha256="TEXT", fingerprint="TEXT", ekey="TEXT", url_key="TEXT", forced="INTEGER DEFAULT 0",
                transcript_id="TEXT", cached="INTEGER DEFAULT 0", audio_s="REAL")


def _iso(t):
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def is_url(s): return str(s).startswith(("http://", "https://"))


def fingerprint(p, words=None):
    d = dict(model=MODEL_DEFAULT, chunk_minutes=float(p["chunk_minutes"]), overlap_seconds=float(p["overlap_seconds"]),
             words=bool(p["words"] if words is None else words), pipeline_version=PIPELINE_VERSION)
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fingerprints(p):
    """Fingerprints whose transcript satisfies a request with params `p` (a words=true result also serves words=false)."""
    return [fingerprint(p)] + ([fingerprint(p, True)] if not p["words"] else [])


def url_ident(u):
    s = urlsplit(u.strip()); host = (s.hostname or "").lower()
    port = f":{s.port}" if s.port and s.port != {"http": 80, "https": 443}.get(s.scheme.lower()) else ""
    return "url:" + urlunsplit((s.scheme.lower(), host + port, s.path or "/", s.query, ""))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def dirsize(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs if os.path.exists(os.path.join(d, f)))


class Store:
    """Thread-safe SQLite tables: jobs, transcripts (cached results), tkeys (client keys / normalized URLs -> transcript)."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT, progress REAL,
            message TEXT, error TEXT, source TEXT NOT NULL, params TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL)""")
        have = [c["name"] for c in self.db.execute("PRAGMA table_info(jobs)")]
        for c, t in NEW_COLS.items():  # migration for older DBs
            if c not in have: self.db.execute(f"ALTER TABLE jobs ADD COLUMN {c} {t}")
        self.db.execute("""CREATE TABLE IF NOT EXISTS transcripts (id TEXT PRIMARY KEY, sha256 TEXT, fingerprint TEXT NOT NULL, name TEXT,
            audio_s REAL, size_bytes INTEGER, created_at REAL NOT NULL, last_used_at REAL NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS transcripts_sha ON transcripts(sha256)")
        self.db.execute("CREATE TABLE IF NOT EXISTS tkeys (ekey TEXT NOT NULL, transcript_id TEXT NOT NULL, PRIMARY KEY (ekey, transcript_id))")
        self.db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS jobs_active_fp ON jobs(sha256, fingerprint)
            WHERE sha256 IS NOT NULL AND forced=0 AND status IN ('queued','running')""")

    def q(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def execute(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).rowcount

    def get(self, id):
        r = self.q(f"SELECT {COLS} FROM jobs WHERE id=?", (id,))
        return r[0] if r else None

    def insert(self, **c):
        self.execute(f"INSERT INTO jobs ({','.join(c)}) VALUES ({','.join('?' * len(c))})", tuple(c.values()))

    def transcript(self, tid):
        r = self.q("SELECT * FROM transcripts WHERE id=?", (tid,))
        return r[0] if r else None

    def find_transcript(self, sha, idents, fps):
        ph = ",".join("?" * len(fps))
        if sha: r = self.q(f"SELECT * FROM transcripts WHERE sha256=? AND fingerprint IN ({ph}) ORDER BY created_at DESC LIMIT 1", (sha, *fps))
        elif idents:
            r = self.q(f"SELECT t.* FROM transcripts t JOIN tkeys k ON k.transcript_id=t.id WHERE k.ekey IN ({','.join('?' * len(idents))}) "
                       f"AND t.fingerprint IN ({ph}) ORDER BY t.created_at DESC LIMIT 1", (*idents, *fps))
        else: r = []
        return r[0] if r else None

    def find_active(self, sha, idents, fps):
        ph = ",".join("?" * len(fps)); conds, args = [], []
        if sha: conds.append("sha256=?"); args.append(sha)
        if idents:
            qm = ",".join("?" * len(idents)); conds.append(f"ekey IN ({qm}) OR url_key IN ({qm})"); args += idents * 2
        if not conds: return None
        r = self.q(f"SELECT {COLS} FROM jobs WHERE {ACTIVE} AND fingerprint IN ({ph}) AND ({' OR '.join(conds)}) ORDER BY created_at LIMIT 1", (*fps, *args))
        return r[0] if r else None

    def touch(self, tid): self.execute("UPDATE transcripts SET last_used_at=? WHERE id=?", (time.time(), tid))

    def add_keys(self, tid, *keys):
        for k in keys:
            if k: self.execute("INSERT OR IGNORE INTO tkeys (ekey, transcript_id) VALUES (?,?)", (k, tid))


def overall_progress(r):
    if r["status"] == "done": return 1.0
    if r["status"] == "queued" or r["stage"] not in STAGES: return 0.0
    lo, hi = STAGES[r["stage"]]
    return round(lo + (hi - lo) * min(max(r["progress"] or 0.0, 0.0), 1.0), 4)


def public(r, store):
    pos = None
    if r["status"] == "queued":
        pos = store.q("SELECT COUNT(*) n FROM jobs WHERE status='running' OR (status='queued' AND created_at<?)", (r["created_at"],))[0]["n"]
    return dict(id=r["id"], status=r["status"], stage=r["stage"], progress=r["progress"], message=r["message"], error=r["error"],
                params=json.loads(r["params"]), created_at=_iso(r["created_at"]), started_at=_iso(r["started_at"]),
                finished_at=_iso(r["finished_at"]), overall_progress=overall_progress(r), queue_position=pos,
                transcript_id=r["transcript_id"] if r["status"] == "done" else None, cached=bool(r["cached"]),
                audio_s=r["audio_s"], sha256=r["sha256"], key=r["ekey"])


def default_run(*a, **kw):
    from transcriber.pipeline import run  # lazy: keeps app import light
    return run(*a, **kw)


def default_fetch(url, work):
    from transcriber.pipeline import get_audio
    os.makedirs(work, exist_ok=True)
    return get_audio(url, work)


class Service:
    def __init__(self, state_dir, run_fn=None, start_worker=True, fetch_fn=None, cache_max_gb=None, cache_ttl_days=None):
        self.state = os.path.abspath(state_dir)
        self.uploads = os.path.join(self.state, "uploads"); self.jobs = os.path.join(self.state, "jobs")
        self.transcripts = os.path.join(self.state, "transcripts"); self.chunks = os.path.join(self.state, "cache", "chunks")
        for d in (self.uploads, self.jobs, self.transcripts, self.chunks): os.makedirs(d, exist_ok=True)
        self.max_bytes = float(cache_max_gb if cache_max_gb is not None else os.environ.get("CACHE_MAX_GB") or 20) * 1e9
        ttl = cache_ttl_days if cache_ttl_days is not None else os.environ.get("CACHE_TTL_DAYS")
        self.ttl_days = float(ttl) if ttl else None
        self.store = Store(os.path.join(self.state, "jobs.db"))
        self.run_fn = run_fn or default_run; self.fetch_fn = fetch_fn or default_fetch
        self.queue = queue.Queue(); self.stop = threading.Event(); self.thread = None
        self.migrate()
        # crash/restart recovery: running jobs resume (chunk cache makes that cheap), queued jobs keep their order
        self.store.execute("UPDATE jobs SET status='queued', stage='queued', message='re-queued after restart', started_at=NULL WHERE status='running'")
        for r in self.store.q("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"): self.queue.put(r["id"])
        if start_worker: self.start()

    def migrate(self):
        """Jobs from the pre-transcript schema: fill fingerprint/url_key; finished ones with result files become transcripts (sha256 unknown)."""
        for r in self.store.q("SELECT id,status,source,params,finished_at,created_at FROM jobs WHERE fingerprint IS NULL"):
            p = json.loads(r["params"]); jd = self.job_dir(r["id"]); tid = None
            self.store.execute("UPDATE jobs SET fingerprint=?, url_key=? WHERE id=?", (fingerprint(p), url_ident(r["source"]) if is_url(r["source"]) else None, r["id"]))
            if r["status"] != "done" or not os.path.exists(os.path.join(jd, FORMATS["json"][0])): continue
            tid = r["id"]; dest = os.path.join(self.transcripts, tid); os.makedirs(dest, exist_ok=True)
            for n, _ in FORMATS.values():
                if os.path.exists(os.path.join(jd, n)): shutil.move(os.path.join(jd, n), os.path.join(dest, n))
            shutil.rmtree(jd, ignore_errors=True)
            t = r["finished_at"] or r["created_at"]
            self.store.execute("INSERT OR IGNORE INTO transcripts (id,sha256,fingerprint,name,audio_s,size_bytes,created_at,last_used_at) VALUES (?,NULL,?,?,?,?,?,?)",
                               (tid, fingerprint(p), self.store.q("SELECT name FROM jobs WHERE id=?", (tid,))[0]["name"], self.audio_s(dest), dirsize(dest), t, t))
            self.store.execute("UPDATE jobs SET transcript_id=?, audio_s=? WHERE id=?", (tid, self.audio_s(dest), tid))
            if is_url(r["source"]): self.store.add_keys(tid, url_ident(r["source"]))

    @staticmethod
    def audio_s(d):
        try: return json.load(open(os.path.join(d, "stats.json")))["audio_s"]
        except (OSError, ValueError, KeyError, TypeError): return None

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="transcribe-worker", daemon=True); self.thread.start()

    def shutdown(self, timeout=15.0):
        """Stop taking jobs, kill the active pipeline child; the interrupted job is put back to `queued` (see _run)."""
        self.stop.set(); self.queue.put(None)
        from transcriber.pipeline import kill_children
        kill_children()
        if self.thread is not None and self.thread is not threading.current_thread(): self.thread.join(timeout)

    def submit(self, source, params, id=None, name=None, sha256=None, key=None, force=False):
        """Atomic dedupe + insert. Returns dict(id, status, cached[, transcript_id]); cached=True means no new job was queued
        (either a finished transcript was linked to a new done job, or the id of the matching active job is returned)."""
        id = id or uuid.uuid4().hex[:12]; fps = fingerprints(params); url = is_url(source)
        url_key = url_ident(source) if url else None
        idents = [k for k in (key, url_key) if k] if url else []
        with self.store.lock:
            if not force:
                t = self.store.find_transcript(sha256, idents, fps)
                if t is not None:
                    self.store.insert(id=id, status="done", stage="done", progress=1.0, message="cached", source=source, params=json.dumps(params),
                                      created_at=time.time(), finished_at=time.time(), name=name, sha256=t["sha256"] or sha256, fingerprint=t["fingerprint"],
                                      ekey=key, url_key=url_key, transcript_id=t["id"], cached=1, audio_s=t["audio_s"])
                    self.store.touch(t["id"]); self.store.add_keys(t["id"], *idents, *([key] if key else []))
                    if not url: self._rm(source)
                    return dict(id=id, status="done", cached=True, transcript_id=t["id"])
                a = self.store.find_active(sha256, idents, fps)
                if a is not None:
                    if not url: self._rm(source)
                    return dict(id=a["id"], status=a["status"], cached=True)
            elif sha256 and not self.store.q(f"SELECT 1 FROM jobs WHERE {ACTIVE} AND sha256=?", (sha256,)):
                shutil.rmtree(os.path.join(self.chunks, sha256), ignore_errors=True)  # force: recompute from scratch
            self.store.insert(id=id, status="queued", stage="queued", source=source, params=json.dumps(params), created_at=time.time(), name=name,
                              sha256=sha256, fingerprint=fingerprint(params), ekey=key, url_key=url_key, forced=int(bool(force)))
        self.queue.put(id)
        return dict(id=id, status="queued", cached=False)

    @staticmethod
    def _rm(path):
        try: os.remove(path)
        except OSError: pass

    def job_dir(self, id): return os.path.join(self.jobs, id)

    def remove_files(self, id):
        shutil.rmtree(self.job_dir(id), ignore_errors=True)
        for f in os.listdir(self.uploads):
            if f.split(".")[0] == id: os.remove(os.path.join(self.uploads, f))

    def drop_transcript(self, tid):
        with self.store.lock:
            self.store.execute("DELETE FROM transcripts WHERE id=?", (tid,)); self.store.execute("DELETE FROM tkeys WHERE transcript_id=?", (tid,))
            self.store.execute("UPDATE jobs SET transcript_id=NULL WHERE transcript_id=?", (tid,))
        shutil.rmtree(os.path.join(self.transcripts, tid), ignore_errors=True)

    def evict(self, keep_tid=None, keep_sha=None):
        """Drop least recently used transcripts / chunk caches until under max_bytes (and anything older than the TTL).
        Entries whose audio sha256 belongs to an active job, and the keep_* ones, are never evicted. Returns how many were removed."""
        with self.store.lock:
            now = time.time()
            active = {r["sha256"] for r in self.store.q(f"SELECT sha256 FROM jobs WHERE {ACTIVE} AND sha256 IS NOT NULL")}
            ents = [(r["last_used_at"], r["size_bytes"] or 0, "t", r["id"], r["sha256"] in active or r["id"] == keep_tid)
                    for r in self.store.q("SELECT id,sha256,size_bytes,last_used_at FROM transcripts")]
            for n in os.listdir(self.chunks):
                p = os.path.join(self.chunks, n); ents.append((os.path.getmtime(p), dirsize(p), "c", n, n in active or n == keep_sha))
            total = sum(e[1] for e in ents); n_removed = 0
            for lu, sz, kind, ident, prot in sorted(ents):
                if prot or not (total > self.max_bytes or (self.ttl_days and lu < now - self.ttl_days * 86400)): continue
                if kind == "t": self.drop_transcript(ident)
                else: shutil.rmtree(os.path.join(self.chunks, ident), ignore_errors=True)
                total -= sz; n_removed += 1
            return n_removed

    def _loop(self):
        while not self.stop.is_set():
            id = self.queue.get()
            if id is None: break
            # claim atomically; skips jobs cancelled/deleted while queued
            if self.store.execute("UPDATE jobs SET status='running', stage='starting', started_at=?, finished_at=NULL WHERE id=? AND status='queued'", (time.time(), id)) != 1:
                continue
            self._run(id)

    def _link_cached(self, id, sha, fps, upload):
        with self.store.lock:
            t = self.store.find_transcript(sha, [], fps)
            if t is None: return False
            self.store.execute("UPDATE jobs SET status='done', stage='done', progress=1.0, message='cached', transcript_id=?, cached=1, audio_s=?, finished_at=? WHERE id=? AND status='running'",
                               (t["id"], t["audio_s"], time.time(), id))
            self.store.touch(t["id"])
        if upload: self._rm(upload)
        return True

    def _publish(self, id, r, p, sha):
        tid = uuid.uuid4().hex[:12]; src = self.job_dir(id); dest = os.path.join(self.transcripts, tid); os.makedirs(dest)
        for n, _ in FORMATS.values():
            if os.path.exists(os.path.join(src, n)): shutil.move(os.path.join(src, n), os.path.join(dest, n))
        shutil.rmtree(src, ignore_errors=True)
        fp = fingerprint(p); now = time.time(); aud = self.audio_s(dest)
        with self.store.lock:
            old = self.store.q("SELECT id FROM transcripts WHERE sha256=? AND fingerprint=?", (sha, fp)) if sha else []  # superseded by this one
            self.store.execute("INSERT INTO transcripts (id,sha256,fingerprint,name,audio_s,size_bytes,created_at,last_used_at) VALUES (?,?,?,?,?,?,?,?)",
                               (tid, sha, fp, r["name"], aud, dirsize(dest), now, now))
            self.store.add_keys(tid, r["ekey"], url_ident(r["source"]) if is_url(r["source"]) else None)
            for o in old:
                self.store.execute("UPDATE OR IGNORE tkeys SET transcript_id=? WHERE transcript_id=?", (tid, o["id"]))
                self.store.execute("UPDATE jobs SET transcript_id=? WHERE transcript_id=?", (tid, o["id"]))
                self.drop_transcript(o["id"])
            self.store.execute("UPDATE jobs SET status='done', stage='done', progress=1.0, message='finished', transcript_id=?, audio_s=?, finished_at=? WHERE id=? AND status='running'",
                               (tid, aud, now, id))
        return tid

    def _run(self, id):
        r = self.store.get(id); p = json.loads(r["params"]); outdir = self.job_dir(id); dl = os.path.join(outdir, "dl")
        upload = not is_url(r["source"]); tid = sha = None
        def progress(stage, fraction, message):
            self.store.execute("UPDATE jobs SET stage=?, progress=?, message=? WHERE id=? AND status='running'", (stage, fraction, message, id))
        try:
            src = r["source"]
            if not upload:
                progress("prepare", None, "downloading audio")
                src = self.fetch_fn(src, dl)
            sha = r["sha256"] if upload and r["sha256"] else sha256_file(src)
            try: self.store.execute("UPDATE jobs SET sha256=? WHERE id=?", (sha, id))
            except sqlite3.IntegrityError: pass  # another active job already owns (sha, settings); this one still runs
            if not r["forced"] and self._link_cached(id, sha, fingerprints(p), r["source"] if upload else None): return
            cache_dir = os.path.join(self.chunks, sha); os.makedirs(cache_dir, exist_ok=True); os.utime(cache_dir)
            self.run_fn(src, outdir, chunk_minutes=p["chunk_minutes"], overlap_seconds=p["overlap_seconds"], context=p["context"],
                        words=p["words"], progress=progress, input_name=r["name"], cache_dir=cache_dir)
            os.utime(cache_dir)
            tid = self._publish(id, r, p, sha)
            if upload: self._rm(r["source"])  # upload no longer needed once done
        except BaseException as e:
            if self.stop.is_set():  # interrupted by service shutdown (SIGTERM): resume on next start
                self.store.execute("UPDATE jobs SET status='queued', stage='queued', message='re-queued after shutdown', started_at=NULL WHERE id=? AND status='running'", (id,))
                return
            traceback.print_exc()
            msg = str(e) or type(e).__name__
            self.store.execute("UPDATE jobs SET status='failed', error=?, message=?, finished_at=? WHERE id=? AND status='running'", (msg, msg, time.time(), id))
        finally:
            shutil.rmtree(dl, ignore_errors=True)
        try: self.evict(keep_tid=tid, keep_sha=sha)
        except Exception: traceback.print_exc()


def _num(v, name, default, lo, hi, hi_incl=True):
    if v in (None, ""): return default
    try: x = float(v)
    except (TypeError, ValueError): raise HTTPException(400, f"{name} must be a number")
    if not (lo < x <= hi if hi_incl else lo < x < hi): raise HTTPException(400, f"{name} out of range")
    return x


def _bool(v, default, name="words"):
    if v in (None, ""): return default
    if isinstance(v, bool): return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"): return True
    if s in ("0", "false", "no", "off"): return False
    raise HTTPException(400, f"{name} must be a boolean")


def create_app(state_dir=None, run_fn=None, api_token=None, start_worker=True, fetch_fn=None, cache_max_gb=None, cache_ttl_days=None):
    state_dir = state_dir or os.environ.get("STATE_DIR") or os.path.expanduser("~/.local/state/transcribe-service")
    api_token = api_token if api_token is not None else (os.environ.get("API_TOKEN") or None)
    svc = Service(state_dir, run_fn, start_worker, fetch_fn, cache_max_gb, cache_ttl_days)

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
    atexit.register(svc.shutdown, 5.0)  # last resort if the process exits without the lifespan running

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "queued": len(svc.store.q("SELECT 1 FROM jobs WHERE status='queued'")),
                "running": len(svc.store.q("SELECT 1 FROM jobs WHERE status='running'"))}

    @app.post("/jobs", dependencies=[Depends(auth)])
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
        force = _bool(data.get("force"), False, "force")
        key = str(data.get("key") or "").strip()[:512] or None
        want = str(data.get("sha256") or "").strip().lower() or None
        if want and not re.fullmatch(r"[0-9a-f]{64}", want): raise HTTPException(400, "sha256 must be 64 lowercase hex characters")
        id = uuid.uuid4().hex[:12]; sha = None
        if upload is not None:
            ext = re.sub(r"[^A-Za-z0-9.]", "", os.path.splitext(upload.filename or "")[1])[:8]
            source = os.path.join(svc.uploads, id + ext)
            def save():
                h = hashlib.sha256()
                with open(source, "wb") as f:
                    for b in iter(lambda: upload.file.read(1 << 20), b""): h.update(b); f.write(b)
                return h.hexdigest()
            sha = await run_in_threadpool(save)
            await upload.close()
            if want and want != sha:
                svc._rm(source); raise HTTPException(400, f"sha256 mismatch: uploaded content hashes to {sha}")
        else:
            source = str(url)
        res = svc.submit(source, params, id, name=(upload.filename if upload is not None else source), sha256=sha, key=key, force=force)
        if res["cached"]: return JSONResponse(res, status_code=200)
        return JSONResponse(res, status_code=202)

    def need(id):
        r = svc.store.get(id)
        if r is None: raise HTTPException(404, "no such job")
        return r

    def send(tid, format):
        if format not in FORMATS: raise HTTPException(400, f"format must be one of {sorted(FORMATS)}")
        name, mt = FORMATS[format]; path = os.path.join(svc.transcripts, tid, name)
        if not os.path.exists(path): raise HTTPException(404, f"{name} missing")
        svc.store.touch(tid)
        return FileResponse(path, media_type=mt)

    @app.get("/jobs", dependencies=[Depends(auth)])
    def list_jobs():
        return [public(r, svc.store) for r in svc.store.q(f"SELECT {COLS} FROM jobs ORDER BY created_at DESC")]

    @app.get("/jobs/{id}", dependencies=[Depends(auth)])
    def get_job(id: str):
        return public(need(id), svc.store)

    @app.get("/jobs/{id}/result", dependencies=[Depends(auth)])
    def result(id: str, format: str = "json"):
        r = need(id)
        if format not in FORMATS: raise HTTPException(400, f"format must be one of {sorted(FORMATS)}")
        if r["status"] != "done": raise HTTPException(409, f"job is {r['status']}, not done")
        if not r["transcript_id"]: raise HTTPException(404, "transcript no longer cached")
        return send(r["transcript_id"], format)

    @app.delete("/jobs/{id}", dependencies=[Depends(auth)])
    def delete(id: str):
        r = need(id)
        if r["status"] == "running": raise HTTPException(409, "job is running and cannot be cancelled")
        svc.store.execute("DELETE FROM jobs WHERE id=? AND status!='running'", (id,))
        svc.remove_files(id)  # the transcript stays cached
        return {"id": id, "deleted": True, "was": r["status"]}

    @app.get("/transcripts", dependencies=[Depends(auth)])
    def lookup(sha256: str = None, key: str = None):
        sha = (sha256 or "").strip().lower() or None; key = (key or "").strip() or None
        if not sha and not key: raise HTTPException(400, "provide `sha256` or `key`")
        fps = [fingerprint(DEFAULT_PARAMS)]; idents = [] if sha else [key]
        t = svc.store.find_transcript(sha, idents, fps)
        if t is not None:
            svc.store.touch(t["id"])
            j = svc.store.q("SELECT id FROM jobs WHERE transcript_id=? ORDER BY created_at DESC LIMIT 1", (t["id"],))
            ks = svc.store.q("SELECT ekey FROM tkeys WHERE transcript_id=? AND ekey NOT LIKE 'url:%' LIMIT 1", (t["id"],))
            return dict(status="done", transcript_id=t["id"], job_id=j[0]["id"] if j else None, audio_s=t["audio_s"], sha256=t["sha256"],
                        key=key or (ks[0]["ekey"] if ks else None))
        a = svc.store.find_active(sha, idents, fps)
        if a is not None:
            return dict(status=a["status"], job_id=a["id"], transcript_id=None, sha256=a["sha256"] or sha, key=key or a["ekey"])
        raise HTTPException(404, "no transcript or active job for that sha256/key")

    @app.get("/transcripts/{tid}/result", dependencies=[Depends(auth)])
    def transcript_result(tid: str, format: str = "json"):
        if svc.store.transcript(tid) is None: raise HTTPException(404, "no such transcript")
        return send(tid, format)

    @app.delete("/transcripts/{tid}", dependencies=[Depends(auth)])
    def delete_transcript(tid: str):
        t = svc.store.transcript(tid)
        if t is None: raise HTTPException(404, "no such transcript")
        with svc.store.lock:
            if t["sha256"] and svc.store.q(f"SELECT 1 FROM jobs WHERE {ACTIVE} AND sha256=?", (t["sha256"],)):
                raise HTTPException(409, "an active job references this transcript")
            svc.drop_transcript(tid)
        return {"id": tid, "deleted": True}

    return app


def main():
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
