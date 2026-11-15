import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from joons_listen import config, jobs, pipeline, server

RESULT = {
    "engine": "fake", "language": "ko", "source": {"path": "a.wav", "duration_sec": 5, "sha256": "0" * 64},
    "speakers": [{"id": "S1", "label": "S1"}],
    "segments": [{"i": 0, "start": 0.0, "end": 1.0, "speaker": "S1", "text": "안녕하세요", "flags": []}],
    "facts": [], "uncertain": [], "warnings": ["원래 경고"],
}


class CheckAudioPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_srv_")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "inbox"
        self.root.mkdir()
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        p = mock.patch.object(config, "ALLOWED_ROOTS", [self.root])
        p.start()
        self.addCleanup(p.stop)

    def code(self, raw):
        with self.assertRaises(server.ApiError) as cm:
            server.check_audio_path(raw)
        return cm.exception.status, cm.exception.code

    def test_ok(self):
        f = self.root / "회의.M4A"
        f.write_bytes(b"x")
        self.assertEqual(server.check_audio_path(str(f)), f.resolve())

    def test_bad_input(self):
        for raw in (None, "", 123):
            self.assertEqual(self.code(raw), (400, "BAD_REQUEST"))
        self.assertEqual(self.code("relative/a.wav"), (400, "BAD_REQUEST"))
        self.assertEqual(self.code("nofile_anywhere_xyz.wav"), (404, "FILE_NOT_FOUND"))

    def test_outside_root(self):
        f = self.outside / "a.wav"
        f.write_bytes(b"x")
        self.assertEqual(self.code(str(f)), (403, "PATH_NOT_ALLOWED"))

    def test_prefix_sibling_not_allowed(self):
        sib = self.tmp / "inbox_evil"
        sib.mkdir()
        (sib / "a.wav").write_bytes(b"x")
        self.assertEqual(self.code(str(sib / "a.wav")), (403, "PATH_NOT_ALLOWED"))

    def test_dotdot_escape(self):
        (self.outside / "a.wav").write_bytes(b"x")
        self.assertEqual(self.code(str(self.root / ".." / "outside" / "a.wav")), (403, "PATH_NOT_ALLOWED"))

    def test_secret_names(self):
        for name in ("my_password.wav", "API_KEY.mp3", "token.m4a", "x.env.wav", "Secret-meeting.wav"):
            (self.root / name).write_bytes(b"x")
            self.assertEqual(self.code(str(self.root / name)), (403, "PATH_NOT_ALLOWED"), name)

    def test_missing_file(self):
        self.assertEqual(self.code(str(self.root / "none.wav")), (404, "FILE_NOT_FOUND"))

    def test_directory_is_not_file(self):
        (self.root / "dir.wav").mkdir()
        self.assertEqual(self.code(str(self.root / "dir.wav")), (404, "FILE_NOT_FOUND"))

    def test_bad_extension(self):
        f = self.root / "a.txt"
        f.write_bytes(b"x")
        self.assertEqual(self.code(str(f)), (400, "BAD_REQUEST"))
        g = self.root / "noext"
        g.write_bytes(b"x")
        self.assertEqual(self.code(str(g)), (400, "BAD_REQUEST"))


class JobManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_jobs_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        data = self.tmp / "data"
        for name, val in (("DATA_DIR", data), ("INBOX_DIR", data / "inbox"), ("JOBS_DIR", data / "jobs"), ("MODELS_DIR", data / "models"), ("MIRROR_MODE", "off")):
            p = mock.patch.object(config, name, val)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ)  # ensure_dirs 가 넣는 HF_HOME 등을 되돌린다
        env.start()
        self.addCleanup(env.stop)
        self.db_path = self.tmp / "t.sqlite3"
        self.jobs_dir = self.tmp / "jobs"

    def make(self, runner, **kw):
        m = jobs.JobManager(db_path=self.db_path, jobs_dir=self.jobs_dir, mgr=object(), runner=runner, **kw)
        self.addCleanup(self._close, m)
        return m

    @staticmethod
    def _close(m):
        m.shutdown()
        if m.thread.is_alive():
            m.thread.join(5)
        try:
            m.db.close()
        except Exception:
            pass

    def wait(self, m, jid, states, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            j = m.get(jid)
            if j["state"] in states:
                return j
            time.sleep(0.02)
        self.fail(f"상태 {states} 에 도달하지 못함: {m.get(jid)}")

    def test_queued_running_done(self):
        seen = {}
        gate = threading.Event()

        def runner(ctx, kind):
            seen["kind"] = kind
            seen["params"] = ctx.params
            ctx.set_stage("asr", 0.5)
            ctx.warn("작업 경고")
            gate.wait(5)
            return dict(RESULT)

        done = []
        m = self.make(runner, on_done=lambda j, d: done.append(j))
        jid = m.submit("transcribe", {"audio_path": "C:/a/회의.wav"})
        self.assertEqual(m.get(jid)["state"], "queued")
        self.assertEqual(m.get(jid)["title"], "회의")
        self.assertEqual(m.counts(), {"running": 0, "queued": 1})
        m.start()
        self.wait(m, jid, {"running"})
        self.assertEqual(m.counts()["running"], 1)
        gate.set()
        j = self.wait(m, jid, {"done"})
        self.assertEqual(seen["kind"], "transcribe")
        self.assertEqual(seen["params"]["kind"], "transcribe")
        self.assertEqual(j["progress"], 100.0)
        self.assertEqual(j["warnings"], ["원래 경고", "작업 경고"])
        self.assertEqual(set(j["outputs"]), {"json", "txt", "md", "srt", "minutes"})
        self.assertEqual(j["mirror"], "off")
        self.assertEqual(j["engine"], "fake")
        self.assertIn("안녕하세요", m.result_text(jid, "txt"))
        self.assertIsNone(m.result_text(jid, "nope"))
        self.assertIsNone(m.result_text("lst_none", "txt"))
        self.assertTrue((self.jobs_dir / jid / "result.json").exists())
        self.assertEqual(done, [jid])
        self.assertEqual(m.counts(), {"running": 0, "queued": 0})
        self.assertEqual([x["job_id"] for x in m.list()], [jid])

    def test_mirror_pending_when_enabled(self):
        with mock.patch.object(config, "MIRROR_MODE", "folder"):
            m = self.make(lambda ctx, kind: dict(RESULT))
            jid = m.submit("transcribe", {"audio_path": "a.wav"})
            m.start()
            self.wait(m, jid, {"done"})
            self.assertEqual(m.pending_sync(), [jid])
            m.set_mirror(jid, "ok")
            self.assertEqual(m.pending_sync(), [])

    def test_on_done_exception_ignored(self):
        def boom(j, d):
            raise RuntimeError("x")

        m = self.make(lambda ctx, kind: dict(RESULT), on_done=boom)
        jid = m.submit("transcribe", {"audio_path": "a.wav"})
        m.start()
        self.assertEqual(self.wait(m, jid, {"done", "error"})["state"], "done")

    def test_error_job_error_and_generic(self):
        def runner(ctx, kind):
            if ctx.params["audio_path"] == "bad":
                raise pipeline.JobError("AUDIO_BAD", "깨진 파일")
            raise ValueError("뜻밖")

        m = self.make(runner)
        a = m.submit("transcribe", {"audio_path": "bad"})
        b = m.submit("transcribe", {"audio_path": "other"})
        m.start()
        ja = self.wait(m, a, {"error"})
        jb = self.wait(m, b, {"error"})
        self.assertEqual(ja["error"], {"code": "AUDIO_BAD", "message": "깨진 파일"})
        self.assertEqual(jb["error"]["code"], "INTERNAL")
        self.assertIn("뜻밖", jb["error"]["message"])
        # 오류 뒤에도 작업자가 살아 있어 다음 작업을 처리한다
        m.runner = lambda ctx, kind: dict(RESULT)
        c = m.submit("transcribe", {"audio_path": "ok"})
        self.assertEqual(self.wait(m, c, {"done", "error"})["state"], "done")

    def test_cancel_running(self):
        started = threading.Event()

        def runner(ctx, kind):
            started.set()
            for _ in range(500):
                ctx.check()
                time.sleep(0.01)
            return dict(RESULT)

        m = self.make(runner)
        jid = m.submit("transcribe", {"audio_path": "a.wav"})
        m.start()
        self.assertTrue(started.wait(5))
        self.assertEqual(m.cancel(jid), "running")
        j = self.wait(m, jid, {"cancelled"})
        self.assertEqual(j["state"], "cancelled")
        self.assertNotIn(jid, m.cancel_flags)

    def test_cancel_queued_and_unknown_and_finished(self):
        m = self.make(lambda ctx, kind: dict(RESULT))
        jid = m.submit("transcribe", {"audio_path": "a.wav"})
        self.assertEqual(m.cancel(jid), "cancelled")
        self.assertEqual(m.get(jid)["state"], "cancelled")
        self.assertIsNone(m.cancel("lst_nope"))
        m.start()
        j2 = m.submit("transcribe", {"audio_path": "b.wav"})
        self.wait(m, j2, {"done"})
        self.assertEqual(m.cancel(j2), "done")
        time.sleep(0.1)
        self.assertEqual(m.get(jid)["state"], "cancelled")  # 취소된 대기 작업은 실행되지 않는다

    def test_restart_running_to_queued(self):
        m1 = self.make(lambda ctx, kind: dict(RESULT))
        a = m1.submit("transcribe", {"audio_path": "a.wav"})
        b = m1.submit("transcribe", {"audio_path": "b.wav"})
        c = m1.submit("transcribe", {"audio_path": "c.wav"})
        m1.db.execute("UPDATE jobs SET state='running', stage='asr' WHERE id=?", (a,))
        m1.db.execute("UPDATE jobs SET state='done' WHERE id=?", (b,))
        m1.db.commit()
        m1.db.close()
        m2 = self.make(lambda ctx, kind: dict(RESULT))
        self.assertEqual(m2.get(a)["state"], "queued")
        self.assertIsNone(m2.get(a)["stage"])
        self.assertEqual(m2.get(b)["state"], "done")
        self.assertEqual(m2.get(c)["state"], "queued")
        m2.start()
        self.wait(m2, a, {"done"})
        self.wait(m2, c, {"done"})

    def test_progress_stage_weights(self):
        m = self.make(lambda ctx, kind: dict(RESULT))
        self.assertEqual(m._progress("ingest", 0.0), 0.0)
        self.assertEqual(m._progress("vad", 0.0), 5.0)
        self.assertEqual(m._progress("asr", 0.5), 25 + 27.5)
        self.assertEqual(m._progress("store", 1.0), 99.0)  # 99 로 상한
        self.assertEqual(m._progress("asr", 9), m._progress("asr", 1.0))

    def test_default_runner_summarize_uses_parent_result(self):
        pdir = self.jobs_dir / "lst_parent"
        pdir.mkdir(parents=True)
        (pdir / "result.json").write_text(json.dumps(RESULT, ensure_ascii=False), encoding="utf-8")
        m = self.make(None)
        captured = {}

        def fake(ctx, parent):
            captured["p"] = parent
            return dict(RESULT)

        with mock.patch.object(pipeline, "summarize_existing", fake):
            jid = m.submit("summarize", {"parent_job_id": "lst_parent"})
            m.start()
            self.wait(m, jid, {"done", "error"})
        self.assertEqual(m.get(jid)["state"], "done")
        self.assertEqual(captured["p"]["engine"], "fake")


if __name__ == "__main__":
    unittest.main()
