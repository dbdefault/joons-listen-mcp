"""engine=ensemble through pipeline.run with fake engines (no GPU, no model download).

The job folder is pre-filled with the cached ingest/VAD/turn files that pipeline.run reuses on resume,
so the test only exercises the ASR + merge part.
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import soundfile as sf
except ImportError:  # pragma: no cover
    np = sf = None

from joons_listen import config


class FakeQwen:
    def transcribe_batch(self, clips, language, hints=()):
        out = {0: "我们做超链接发泡，用桃树的料。", 1: "价格一四五零美金。"}
        return [{"text": out[self.next + k], "language": "zh", "confidence": 0.9} for k in range(len(clips))]


class FakeWhisper:
    def __init__(self):
        self.calls = 0

    def transcribe(self, clip, language, hints=()):
        out = ["我们做超临界发泡用陶氏的料", "价格1450美金"][self.calls]
        self.calls += 1
        return {"text": out, "language": "zh", "confidence": 0.8}


class FakeMgr:
    def __init__(self):
        self.q, self.w = FakeQwen(), FakeWhisper()
        self.q.next = 0
        self.used = []

    def use(self, which):
        self.used.append(which)
        return self.q if which == "qwen" else self.w

    def unload_all(self):
        pass


@unittest.skipIf(np is None, "numpy/soundfile not installed")
class PipelineEnsembleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_ens_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        d = self.tmp / "job"
        d.mkdir()
        src = self.tmp / "a.wav"
        sf.write(str(src), np.zeros(16000 * 4, dtype="float32"), 16000)
        shutil.copy(src, d / "audio.wav")
        (d / "ingest.json").write_text(json.dumps({"duration_sec": 4.0, "sha256": "x"}), encoding="utf-8")
        (d / "vad.json").write_text(json.dumps([[0.0, 2.0], [2.0, 4.0]]), encoding="utf-8")
        (d / "turns.json").write_text(json.dumps([{"start": 0.0, "end": 2.0, "speaker": "S1"}, {"start": 2.0, "end": 4.0, "speaker": "S2"}]), encoding="utf-8")
        self.dir, self.src = d, src
        cust = self.tmp / "dicts"
        cust.mkdir()
        (cust / "acme.json").write_text(json.dumps({"terms": [{"term": "陶氏", "aliases": ["超临界"]}], "replace": []}, ensure_ascii=False), encoding="utf-8")
        self.patches = [mock.patch.object(config, "CUSTOMER_DICT_DIR", cust), mock.patch.object(config, "DICT_PATH", self.tmp / "none.json"),
                        mock.patch.object(config, "QWEN_BATCH", 8)]
        for pt in self.patches:
            pt.start()
            self.addCleanup(pt.stop)

    def test_ensemble_merges_and_reports(self):
        from joons_listen import llm, pipeline

        stages = []
        ctx = pipeline.Ctx("job1", self.dir, {"audio_path": str(self.src), "engine": "ensemble", "language": "auto", "customer": "acme"},
                           lambda s, f: stages.append((s, f)), lambda: False, lambda w: None)
        mgr = FakeMgr()
        with mock.patch.object(llm, "unload", lambda *a, **k: None):
            res = pipeline.run(ctx, mgr)
        self.assertEqual(mgr.used, ["qwen", "whisper"])
        self.assertEqual(res["engine"], "ensemble(qwen+whisper)")
        s0, s1 = res["segments"]
        self.assertIn("超临界", s0["text"])
        self.assertIn("陶氏", s0["text"])
        self.assertEqual(s0["alts"]["qwen"], "我们做超链接发泡，用桃树的料。")
        self.assertEqual(s1["text"], "价格1450美金。")
        self.assertEqual(s1["agree"], 1.0)
        self.assertGreaterEqual(res["ensemble"]["picked_secondary"], 1)
        self.assertTrue((self.dir / "asr_whisper.jsonl").exists())
        asr = [f for s, f in stages if s == "asr"]
        self.assertTrue(asr and max(asr) <= 1.0)


if __name__ == "__main__":
    unittest.main()
