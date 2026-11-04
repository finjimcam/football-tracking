"""Run tools/ scripts as background jobs with logs and progress.

Each job is one `python tools/<tool> <args>` subprocess (argv list, never a
shell), run from the repo root like the workflows do. Records live in
.tmp/jobs/<id>.json and output in .tmp/jobs/<id>.log, so history survives a
server restart (jobs that were running then are marked `interrupted`).

Two lanes, each a FIFO worker pool:
  heavy (GPU / full-video passes): 1 at a time, run under
      `nice -n 10 ionice -c3` when those exist, and held while the optional
      busy check (e.g. the media server's `media-busy`, exit 0 = busy) says
      the machine is in use, unless the job was submitted with force=True;
  light (artifact-only compute): 2 at a time.

Progress: a tool may print lines like `PROGRESS 120/4500 tracking players`;
the runner parses the latest one into job["progress"]. Tools that don't
print it just show an indeterminate bar.

A proper persistent queue is issue #19; this is the minimal version the
front end needs.
"""
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)\s*/\s*(\d+)\s*(.*)$")
TERMINAL = {"done", "failed", "cancelled", "interrupted"}
BUSY_RECHECK_S = 30


def parse_progress(line):
    m = PROGRESS_RE.match(line.strip())
    if not m:
        return None
    done, total = int(m.group(1)), int(m.group(2))
    return {"done": done, "total": total, "label": m.group(3).strip()}


class JobRunner:
    def __init__(self, root, python=sys.executable, busy_check=None, heavy_workers=1,
                 light_workers=2, nice_heavy=True):
        self.root = Path(root).resolve()
        self.dir = self.root / ".tmp" / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.python = python
        self.busy_check = busy_check  # argv list or None
        self.lock = threading.RLock()
        self.jobs = {}
        self.procs = {}
        self.queues = {"heavy": queue.Queue(), "light": queue.Queue()}
        self.prefix = []
        if nice_heavy:
            if shutil.which("nice"):
                self.prefix += ["nice", "-n", "10"]
            if shutil.which("ionice"):
                self.prefix += ["ionice", "-c3"]
        self._load_history()
        for lane, n in (("heavy", heavy_workers), ("light", light_workers)):
            for i in range(n):
                threading.Thread(target=self._worker, args=(lane,), daemon=True,
                                 name=f"jobs-{lane}-{i}").start()

    # ---- persistence -------------------------------------------------
    def _record_path(self, job_id):
        return self.dir / f"{job_id}.json"

    def log_path(self, job_id):
        return self.dir / f"{job_id}.log"

    def _save(self, job):
        tmp = self._record_path(job["id"]).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(self._record_path(job["id"]))

    def _load_history(self):
        for p in self.dir.glob("*.json"):
            try:
                job = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("status") not in TERMINAL:
                job["status"] = "interrupted"
                job["note"] = "Server restarted while this job was queued or running"
                job["finished"] = job.get("finished") or time.time()
                self._save(job)
            self.jobs[job["id"]] = job

    # ---- public API --------------------------------------------------
    def submit(self, stage, args, match_id=None, force=False):
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job = {
            "id": job_id, "stage": stage.id, "title": stage.title, "tool": stage.tool,
            "args": list(args), "match_id": match_id,
            "lane": "heavy" if stage.heavy else "light", "force": bool(force),
            "status": "queued", "note": None, "progress": None, "returncode": None,
            "created": time.time(), "started": None, "finished": None,
        }
        with self.lock:
            self.jobs[job_id] = job
            self._save(job)
        self.log_path(job_id).write_text("", encoding="utf-8")
        self.queues[job["lane"]].put(job_id)
        return dict(job)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def list(self, match_id=None, limit=100):
        with self.lock:
            jobs = [dict(j) for j in self.jobs.values()
                    if match_id is None or j.get("match_id") == match_id]
        jobs.sort(key=lambda j: j["created"], reverse=True)
        return jobs[:limit]

    def read_log(self, job_id, offset=0, max_bytes=256_000):
        p = self.log_path(job_id)
        if not p.exists():
            return "", offset
        with open(p, "rb") as f:
            f.seek(max(0, offset))
            data = f.read(max_bytes)
        return data.decode("utf-8", errors="replace"), offset + len(data)

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["status"] in TERMINAL:
                return False
            if job["status"] in ("queued", "waiting"):
                self._finish(job, "cancelled", None)
                return True
            proc = self.procs.get(job_id)
            job["cancel_requested"] = True
        if proc:
            self._kill(proc)
        return True

    def force(self, job_id):
        """Let a job held by the busy check start anyway."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job["status"] not in ("queued", "waiting"):
                return False
            job["force"] = True
            self._save(job)
        return True

    # ---- internals ---------------------------------------------------
    def _update(self, job, **kw):
        with self.lock:
            job.update(kw)
            self._save(job)

    def _finish(self, job, status, returncode):
        self._update(job, status=status, returncode=returncode, finished=time.time())

    def _is_busy(self):
        if not self.busy_check:
            return False
        try:
            return subprocess.run(self.busy_check, capture_output=True, timeout=20).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return True  # can't tell: be polite, same as media-busy itself

    def _kill(self, proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _worker(self, lane):
        q = self.queues[lane]
        while True:
            job_id = q.get()
            job = self.jobs.get(job_id)
            if not job or job["status"] in TERMINAL:
                continue
            if lane == "heavy":
                while not job.get("force") and job["status"] not in TERMINAL and self._is_busy():
                    if job["status"] != "waiting":
                        self._update(job, status="waiting",
                                     note="Waiting: the media server is playing something")
                    for _ in range(BUSY_RECHECK_S):
                        if job.get("force") or job["status"] in TERMINAL:
                            break
                        time.sleep(1)
                if job["status"] in TERMINAL:
                    continue
            try:
                self._run(job, lane)
            except Exception as e:  # never let a worker thread die
                with open(self.log_path(job_id), "a", encoding="utf-8") as log:
                    log.write(f"\n[runner error] {e!r}\n")
                self._finish(job, "failed", None)

    def _run(self, job, lane):
        cmd = [self.python, "-u", str(self.root / "tools" / job["tool"]), *job["args"]]
        if lane == "heavy":
            cmd = self.prefix + cmd
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        with open(self.log_path(job["id"]), "a", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(cmd[len(self.prefix) if lane == "heavy" else 0:]) + "\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            with self.lock:
                self.procs[job["id"]] = proc
            self._update(job, status="running", note=None, started=time.time())
            buf = b""
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                # tqdm-style bars redraw with \r; treat it as a line break
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", errors="replace")
                    if not line.strip():
                        continue
                    prog = parse_progress(line)
                    if prog:
                        self._update(job, progress=prog)
                        continue  # keep progress chatter out of the log
                    log.write(line + "\n")
                log.flush()
            if buf.strip():
                log.write(buf.decode("utf-8", errors="replace") + "\n")
            rc = proc.wait()
            log.write(f"\n[exit {rc}]\n")
        with self.lock:
            self.procs.pop(job["id"], None)
        if job.get("cancel_requested"):
            self._finish(job, "cancelled", rc)
        else:
            self._finish(job, "done" if rc == 0 else "failed", rc)
