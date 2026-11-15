import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from joons_listen import config, jobs, store_sync


class MirrorCommand(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp())
        self.job = self.t / "job"
        self.job.mkdir()
        (self.job / "result.json").write_text("{}", encoding="utf-8")
        self.dest = self.t / "dest"
        self.dest.mkdir()

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def _cmd(self, code="import shutil,sys;shutil.copy(sys.argv[1], sys.argv[2])"):
        return json.dumps([sys.executable, "-c", code, "{file}", str(self.dest)])

    def test_command_sends_zip_then_marker(self):
        with mock.patch.object(config, "MIRROR_CMD", self._cmd()), mock.patch.object(config, "MIRROR_MODE", "cmd"):
            ok, msg = store_sync.push("jx", self.job, self.t)
        self.assertTrue(ok, msg)
        self.assertTrue((self.dest / "jx.zip").exists())
        self.assertTrue((self.dest / "jx.ok").exists())
        self.assertFalse((self.t / "sync" / "jx.zip").exists())  # 성공하면 전송용 사본은 지운다

    def test_command_failure_keeps_bundle_for_retry(self):
        with mock.patch.object(config, "MIRROR_CMD", self._cmd("import sys;sys.exit(3)")), mock.patch.object(config, "MIRROR_MODE", "cmd"):
            ok, msg = store_sync.push("jy", self.job, self.t)
        self.assertFalse(ok)
        self.assertIn("(3)", msg)
        self.assertTrue((self.t / "sync" / "jy.zip").exists())

    def test_bad_command_setting_is_rejected(self):
        with mock.patch.object(config, "MIRROR_CMD", "not json"), mock.patch.object(config, "MIRROR_MODE", "cmd"):
            ok, msg = store_sync.push("jz", self.job, self.t)
        self.assertFalse(ok)
        self.assertIn("JSON array", msg)

    def test_missing_program_is_a_failure_not_a_crash(self):
        with mock.patch.object(config, "MIRROR_CMD", json.dumps([str(self.t / "no_such_program"), "{file}"])), mock.patch.object(config, "MIRROR_MODE", "cmd"):
            ok, msg = store_sync.push("jw", self.job, self.t)
        self.assertFalse(ok)


class OldDatabaseColumn(unittest.TestCase):
    def test_brain_sync_column_is_renamed_and_rows_survive(self):
        t = Path(tempfile.mkdtemp())
        try:
            db = sqlite3.connect(str(t / "j.sqlite3"))
            db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT, state TEXT, stage TEXT, progress REAL, params TEXT, error TEXT, warnings TEXT, outputs TEXT, meta TEXT, title TEXT, created TEXT, updated TEXT, brain_sync TEXT)")
            db.execute("INSERT INTO jobs (id,kind,state,brain_sync) VALUES ('lst_old','transcribe','done','pending')")
            db.commit()
            db.close()
            m = jobs.JobManager(db_path=t / "j.sqlite3", jobs_dir=t / "jobs", runner=lambda ctx, kind: {})
            try:
                self.assertEqual(m.pending_sync(), ["lst_old"])
            finally:
                m.shutdown()
        finally:
            shutil.rmtree(t, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
