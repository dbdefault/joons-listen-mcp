import json
import shutil
import tempfile
import unittest
from pathlib import Path

from joons_listen import outputs


def sample():
    return {
        "title": "시험 회의",
        "source": {"path": "C:/x/회의.m4a", "duration_sec": 3725, "sha256": "ab" * 32},
        "engine": "whisper", "language": "ko", "created_at": "2026-11-15T10:00:00+09:00", "version": "1.0.0",
        "warnings": ["경고 하나"],
        "speakers": [{"id": "S1", "label": "김 팀장"}, {"id": "S2", "label": "S2"}],
        "segments": [
            {"i": 0, "start": 0.0, "end": 3.5, "speaker": "S1", "text": "단가는 800 달러입니다", "flags": ["price"]},
            {"i": 1, "start": 3.5, "end": 7.25, "speaker": "S2", "text": "不太确定", "text_ko": "잘 모르겠습니다", "flags": ["uncertain"]},
        ],
        "facts": [{"type": "price", "text": "800 달러", "segment": 0, "verify": True}],
        "uncertain": [1],
        "minutes": {"title": "주간 회의", "llm": "qwen3:8b", "summary": "요약문", "decisions": ["결정1"], "open_questions": ["질문1"],
                    "actions": [{"who": "김", "what": "견적", "due": "11/20"}]},
    }


class TimeFormatTest(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(outputs.hms(0), "00:00:00")
        self.assertEqual(outputs.hms(59.9), "00:00:59")
        self.assertEqual(outputs.hms(3725), "01:02:05")
        self.assertEqual(outputs.hms(-5), "00:00:00")
        self.assertEqual(outputs.hms(360000), "100:00:00")

    def test_srt_ts_basic(self):
        self.assertEqual(outputs.srt_ts(0), "00:00:00,000")
        self.assertEqual(outputs.srt_ts(3725.5), "01:02:05,500")
        self.assertEqual(outputs.srt_ts(-1), "00:00:00,000")

    def test_srt_ts_rounding_boundary(self):
        for v in (59.9995, 59.9996, 59.99999, 0.9995, 3599.9999):
            s = outputs.srt_ts(v)
            self.assertRegex(s, r"^\d{2}:[0-5]\d:[0-5]\d,\d{3}$", v)
            self.assertNotIn(",1000", s)
        self.assertEqual(outputs.srt_ts(59.9996), "00:01:00,000")
        self.assertEqual(outputs.srt_ts(59.99999), "00:01:00,000")
        self.assertEqual(outputs.srt_ts(3599.9999), "01:00:00,000")

    def test_srt_ts_exact_boundary_value(self):
        # 59.9995 는 이진 부동소수점에서 .99949999... 라 999 또는 1000(올림) 이 될 수 있다. 어느 쪽이든 형식은 유효해야 한다.
        self.assertIn(outputs.srt_ts(59.9995), ("00:00:59,999", "00:01:00,000"))


class FormatTest(unittest.TestCase):
    def setUp(self):
        self.r = sample()

    def test_speaker_label(self):
        self.assertEqual(outputs.speaker_label(self.r, "S1"), "김 팀장")
        self.assertEqual(outputs.speaker_label(self.r, "S9"), "S9")

    def test_txt(self):
        lines = outputs.to_txt(self.r).splitlines()
        self.assertEqual(lines[0], "[00:00:00] 김 팀장: 단가는 800 달러입니다")
        self.assertEqual(lines[1], "[00:00:03] S2: 不太确定 ⚠")
        self.assertEqual(lines[2], "    → 잘 모르겠습니다")
        self.assertEqual(len(lines), 3)

    def test_srt(self):
        blocks = outputs.to_srt(self.r).strip().split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].splitlines(), ["1", "00:00:00,000 --> 00:00:03,500", "김 팀장: 단가는 800 달러입니다"])
        self.assertEqual(blocks[1].splitlines(), ["2", "00:00:03,500 --> 00:00:07,250", "S2: 不太确定", "잘 모르겠습니다"])

    def test_text_ko_same_as_text_not_duplicated(self):
        self.r["segments"][1]["text_ko"] = self.r["segments"][1]["text"]
        self.assertNotIn("→", outputs.to_txt(self.r))
        self.assertEqual(len(outputs.to_srt(self.r).strip().split("\n\n")[1].splitlines()), 3)

    def test_md(self):
        md = outputs.to_md(self.r)
        self.assertTrue(md.startswith("# 시험 회의\n"))
        self.assertIn("`회의.m4a`", md)
        self.assertIn("길이 01:02:05", md)
        self.assertIn("화자 2명", md)
        self.assertIn("- 주의: 경고 하나", md)
        self.assertIn("## 확인이 필요한 구간", md)
        self.assertIn("| price | 800 달러 | 00:00:00 | 김 팀장 |", md)
        self.assertIn("**[00:00:03] S2** ⚠", md)
        self.assertIn("> 잘 모르겠습니다", md)

    def test_md_title_fallback_and_minimal(self):
        r = {"source": {"path": "a/b/녹음1.wav"}, "segments": []}
        self.assertTrue(outputs.to_md(r).startswith("# 녹음1\n"))

    def test_minutes(self):
        m = outputs.minutes_md(self.r)
        self.assertTrue(m.startswith("# 주간 회의 — 회의록"))
        self.assertIn("참석(화자 구분): 김 팀장, S2", m)
        self.assertIn("로컬 LLM qwen3:8b", m)
        for head in ("## 요약", "## 결정사항", "## 미결 질문", "## 후속 조치", "## 숫자·날짜·코드", "## 불확실 구간"):
            self.assertIn(head, m)
        self.assertIn("| 김 | 견적 | 11/20 |", m)
        self.assertIn("- price: **800 달러** — [00:00:00] 김 팀장", m)
        self.assertIn("[00:00:03]", m.split("## 불확실 구간")[1])

    def test_minutes_without_llm(self):
        r = sample()
        r["minutes"] = None
        m = outputs.minutes_md(r)
        self.assertTrue(m.startswith("# 시험 회의 — 회의록"))
        self.assertIn("자동 요약 없음", m)
        self.assertNotIn("## 요약", m)


class WriteAllTest(unittest.TestCase):
    def test_write_all(self):
        tmp = Path(tempfile.mkdtemp(prefix="jl_out_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        d = tmp / "nested" / "job"
        r = sample()
        out = outputs.write_all(d, r)
        self.assertEqual(out, {"json": "result.json", "txt": "transcript.txt", "md": "transcript.md", "srt": "subtitles.srt", "minutes": "minutes.md"})
        for name in out.values():
            self.assertTrue((d / name).read_text(encoding="utf-8").strip())
        self.assertEqual(json.loads((d / "result.json").read_text(encoding="utf-8")), r)
        self.assertEqual(list(d.glob("*.tmp")), [])
        self.assertEqual((d / "transcript.txt").read_text(encoding="utf-8"), outputs.to_txt(r))


if __name__ == "__main__":
    unittest.main()
