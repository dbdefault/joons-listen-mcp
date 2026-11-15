import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from joons_listen import config, store_sync


class BuildBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_sync_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.job = self.tmp / "job"
        self.job.mkdir()
        (self.job / "result.json").write_text('{"a": 1}', encoding="utf-8")
        (self.job / "transcript.txt").write_text("안녕", encoding="utf-8")
        (self.job / "audio.wav").write_bytes(b"RIFF" * 100)
        (self.job / "x.json.tmp").write_text("tmp", encoding="utf-8")
        (self.job / "sub").mkdir()
        (self.job / "sub" / "n.txt").write_text("nested", encoding="utf-8")

    def test_bundle(self):
        out = self.tmp / "out" / "deep"
        zp, okp = store_sync.build_bundle("lst_1", self.job, out)
        self.assertEqual((zp.name, okp.name), ("lst_1.zip", "lst_1.ok"))
        with zipfile.ZipFile(zp) as z:
            self.assertEqual(set(z.namelist()), {"result.json", "transcript.txt", "manifest.json"})
            man = json.loads(z.read("manifest.json"))
            self.assertEqual(man["job_id"], "lst_1")
            self.assertEqual(man["version"], config.VERSION)
            self.assertEqual(set(man["files"]), {"result.json", "transcript.txt"})
            data = (self.job / "result.json").read_bytes()
            self.assertEqual(man["files"]["result.json"], {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            self.assertEqual(z.read("result.json"), data)
        want = hashlib.sha256(zp.read_bytes()).hexdigest()
        self.assertEqual(okp.read_text(encoding="ascii"), want + "\n")

    def test_empty_job_dir(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        zp, okp = store_sync.build_bundle("lst_2", empty, self.tmp / "o2")
        with zipfile.ZipFile(zp) as z:
            self.assertEqual(z.namelist(), ["manifest.json"])
        self.assertTrue(okp.exists())

    def test_push_off(self):
        old = config.MIRROR_MODE
        config.MIRROR_MODE = "off"
        try:
            self.assertEqual(store_sync.push("x", self.job, self.tmp), (True, "off"))
        finally:
            config.MIRROR_MODE = old


if __name__ == "__main__":
    unittest.main()
