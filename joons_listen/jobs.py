"""작업 관리: SQLite 작업 목록 + GPU 작업자 스레드(한 번에 하나) + 취소 + 재시작 복구."""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config, engines, outputs, pipeline


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JobManager:
    def __init__(self, db_path: Path = config.DB_PATH, jobs_dir: Path = config.JOBS_DIR, mgr: Optional[engines.EngineManager] = None, on_done: Optional[Callable[[str, Path], None]] = None, runner=None):
        config.ensure_dirs()
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.mgr = mgr or engines.EngineManager()
        self.on_done = on_done
        self.runner = runner or self._default_runner
        self.cancel_flags: Dict[str, bool] = {}
        self.wake = threading.Event()
        self.stop = threading.Event()
        self._last_write: Dict[str, float] = {}
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, state TEXT, stage TEXT, progress REAL, params TEXT, error TEXT, warnings TEXT, outputs TEXT, meta TEXT, title TEXT, created TEXT, updated TEXT, mirror TEXT)"
            )
            # 서비스가 중간에 꺼졌던 작업은 대기열로 되돌려 체크포인트부터 이어 한다.
            self.db.execute("UPDATE jobs SET state='queued', stage=NULL WHERE state='running'")
            self.db.commit()
        self.thread = threading.Thread(target=self._loop, name="listen-worker", daemon=True)

    # ---- 공개 ----
    def start(self) -> None:
        self.thread.start()
        self.wake.set()

    def shutdown(self) -> None:
        self.stop.set()
        self.wake.set()

    def submit(self, kind: str, params: Dict) -> str:
        jid = f"lst_{datetime.now().strftime('%Y%m%d')}_{secrets.token_hex(3)}"
        params = dict(params, kind=kind)
        title = params.get("title") or Path(str(params.get("audio_path", ""))).stem or jid
        with self.lock:
            self.db.execute("INSERT INTO jobs (id,kind,state,stage,progress,params,error,warnings,outputs,meta,title,created,updated,mirror) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (jid, kind, "queued", None, 0.0, json.dumps(params, ensure_ascii=False), None, "[]", "{}", "{}", title, now_iso(), now_iso(), "off"))
            self.db.commit()
        (self.jobs_dir / jid).mkdir(parents=True, exist_ok=True)
        self.wake.set()
        return jid

    def get(self, jid: str) -> Optional[Dict]:
        with self.lock:
            r = self.db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return self._row(r) if r else None

    def list(self, limit: int = 50) -> List[Dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, brief=True) for r in rows]

    def cancel(self, jid: str) -> Optional[str]:
        with self.lock:
            r = self.db.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()
            if not r:
                return None
            if r["state"] == "queued":
                self.db.execute("UPDATE jobs SET state='cancelled', updated=? WHERE id=?", (now_iso(), jid))
                self.db.commit()
                return "cancelled"
            if r["state"] == "running":
                self.cancel_flags[jid] = True
                return "running"
            return r["state"]

    def counts(self) -> Dict:
        with self.lock:
            rows = self.db.execute("SELECT state, COUNT(*) c FROM jobs WHERE state IN ('running','queued') GROUP BY state").fetchall()
        d = {r["state"]: r["c"] for r in rows}
        return {"running": d.get("running", 0), "queued": d.get("queued", 0)}

    def result_text(self, jid: str, fmt: str) -> Optional[str]:
        j = self.get(jid)
        if not j or not j["outputs"].get(fmt):
            return None
        p = self.jobs_dir / jid / j["outputs"][fmt]
        return p.read_text(encoding="utf-8") if p.exists() else None

    def set_mirror(self, jid: str, state: str) -> None:
        with self.lock:
            self.db.execute("UPDATE jobs SET mirror=?, updated=? WHERE id=?", (state, now_iso(), jid))
            self.db.commit()

    def pending_sync(self) -> List[str]:
        with self.lock:
            return [r["id"] for r in self.db.execute("SELECT id FROM jobs WHERE state='done' AND mirror='pending'").fetchall()]

    # ---- 내부 ----
    def _row(self, r: sqlite3.Row, brief: bool = False) -> Dict:
        base = {"job_id": r["id"], "kind": r["kind"], "state": r["state"], "stage": r["stage"], "progress": round(r["progress"] or 0.0, 1), "title": r["title"], "created_at": r["created"], "updated_at": r["updated"]}
        if brief:
            return base
        meta = json.loads(r["meta"] or "{}")
        err = json.loads(r["error"]) if r["error"] else None
        if isinstance(err, dict):
            err.pop("trace", None)  # 내부 추적 정보는 API 로 내보내지 않는다
        base.update({"error": err, "outputs": json.loads(r["outputs"] or "{}"), "warnings": json.loads(r["warnings"] or "[]"), "mirror": r["mirror"]})
        base.update(meta)
        return base

    def _update(self, jid: str, **cols) -> None:
        sets = ",".join(f"{k}=?" for k in cols) + ",updated=?"
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*cols.values(), now_iso(), jid))
            self.db.commit()

    def _progress(self, stage: str, frac: float) -> float:
        done = 0.0
        for s in config.STAGES:
            if s == stage:
                return min(99.0, done + config.STAGE_WEIGHTS[s] * max(0.0, min(1.0, frac)))
            done += config.STAGE_WEIGHTS[s]
        return done

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                job = None
                with self.lock:
                    r = self.db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
                    if r:
                        job = r["id"]
                        self.db.execute("UPDATE jobs SET state='running', updated=? WHERE id=?", (now_iso(), job))
                        self.db.commit()
                if not job:
                    self.wake.wait(timeout=2.0)
                    self.wake.clear()
                    continue
                self._run_one(job)
            except Exception:  # noqa: BLE001  작업자는 어떤 예외로도 죽지 않는다(한 건 실패가 큐 전체를 멈추면 안 됨)
                try:
                    if job:
                        self._update(job, state="error", error=json.dumps({"code": "INTERNAL", "message": "internal worker error"}, ensure_ascii=False))
                except Exception:  # noqa: BLE001
                    pass
                self.stop.wait(1.0)

    def _default_runner(self, ctx: pipeline.Ctx, kind: str) -> Dict:
        if kind == "summarize":
            parent = json.loads((self.jobs_dir / ctx.params["parent_job_id"] / "result.json").read_text(encoding="utf-8"))
            return pipeline.summarize_existing(ctx, parent)
        return pipeline.run(ctx, self.mgr)

    def _run_one(self, jid: str) -> None:
        with self.lock:
            prow = self.db.execute("SELECT kind, params FROM jobs WHERE id=?", (jid,)).fetchone()
        params = json.loads(prow["params"])
        kind = prow["kind"]
        jdir = self.jobs_dir / jid
        jdir.mkdir(parents=True, exist_ok=True)
        warns: List[str] = []

        def set_stage(stage: str, frac: float) -> None:
            t = time.monotonic()
            if frac not in (0.0, 1.0) and t - self._last_write.get(jid, 0) < 0.4:
                return
            self._last_write[jid] = t
            self._update(jid, stage=stage, progress=self._progress(stage, frac))

        ctx = pipeline.Ctx(jid, jdir, params, set_stage, lambda: self.cancel_flags.get(jid, False), warns.append)
        t0 = time.time()
        try:
            result = self.runner(ctx, kind)
            result["warnings"] = list(dict.fromkeys(result.get("warnings", []) + warns))
            outs = outputs.write_all(jdir, result)
            meta = {"engine": result.get("engine"), "language": result.get("language"), "duration_sec": result.get("source", {}).get("duration_sec"), "speakers": len(result.get("speakers", [])), "elapsed_sec": round(time.time() - t0, 1)}
            self._update(jid, state="done", stage="store", progress=100.0, outputs=json.dumps(outs), meta=json.dumps(meta, ensure_ascii=False), warnings=json.dumps(result["warnings"], ensure_ascii=False), mirror="pending" if config.MIRROR_MODE != "off" else "off")
            if self.on_done:
                try:
                    self.on_done(jid, jdir)
                except Exception:  # noqa: BLE001
                    pass
        except pipeline.Cancelled:
            self._update(jid, state="cancelled", stage=None)
        except (pipeline.JobError, engines.EngineError) as e:
            self._update(jid, state="error", error=json.dumps({"code": e.code, "message": e.message}, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            code = getattr(e, "code", "INTERNAL")
            self._update(jid, state="error", error=json.dumps({"code": code if isinstance(code, str) else "INTERNAL", "message": str(e)[:300], "trace": traceback.format_exc()[-800:]}, ensure_ascii=False))
        finally:
            self.cancel_flags.pop(jid, None)
            self._last_write.pop(jid, None)
