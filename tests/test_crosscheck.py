import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from joons_listen import config, crosscheck, llm, pipeline


def seg(i, start, text, conflicts=None):
    return {"i": i, "start": start, "end": start + 5, "speaker": "S1", "text": text, "conflicts": conflicts or []}


class NormalizeTest(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(crosscheck.normalize("价格一千四百五十美金"), "价格1450美金")
        self.assertEqual(crosscheck.normalize("密度零点九六五，二十二点四吨"), "密度0.965224吨".replace("224", "22.4"))
        self.assertEqual(crosscheck.normalize("六五零六 ME-1000"), "6506me1000")


class ReferenceVoteTest(unittest.TestCase):
    def test_switch_confirm_undecided(self):
        segs = [
            seg(0, 0, "牌号是2600的", [{"primary": "2600", "secondary": "2500", "pick": "primary"}]),
            seg(1, 5, "我们用陶氏的料", [{"primary": "陶氏", "secondary": "桃树", "pick": "primary"}]),
            seg(2, 10, "今天天气很好", [{"primary": "天气", "secondary": "天汽", "pick": "primary"}]),
        ]
        refs = [{"name": "doubao", "text": "牌号是2500的。我们用陶氏的料。"}]
        st = crosscheck.reference_vote(segs, refs)
        self.assertEqual(segs[0]["text"], "牌号是2500的")
        self.assertEqual(segs[0]["conflicts"][0]["pick"], "secondary")
        self.assertEqual(segs[0]["conflicts"][0]["resolved_by"], "reference:doubao")
        self.assertEqual(segs[1]["text"], "我们用陶氏的料")
        self.assertEqual(st, {"checked": 3, "confirmed": 1, "switched": 1, "undecided": 1})

    def test_switch_back_to_primary(self):
        segs = [seg(0, 0, "我们用TPU", [{"primary": "TPE", "secondary": "TPU", "pick": "secondary"}])]
        crosscheck.reference_vote(segs, [{"name": "kimi", "text": "我们用TPE做的"}])
        self.assertEqual(segs[0]["text"], "我们用TPE")


class LlmArbitrateTest(unittest.TestCase):
    def test_picks_and_bad_answer(self):
        segs = [seg(0, 0, "做超链接发泡", [{"primary": "超链接", "secondary": "超临界", "pick": "primary"}]),
                seg(1, 5, "今天天气好", [{"primary": "天气", "secondary": "天汽", "pick": "primary"}]),
                seg(2, 10, "随便", [{"primary": "随便", "secondary": "随变", "pick": "primary", "resolved_by": "reference:x"}])]
        seen = []

        def chat(msgs):
            seen.append(msgs[1]["content"])
            return json.dumps({"answers": [{"id": 0, "pick": "B"}, {"id": 1, "pick": "A"}]})

        st = crosscheck.llm_arbitrate(segs, ["超临界"], chat)
        self.assertEqual(segs[0]["text"], "做超临界发泡")
        self.assertEqual(segs[1]["text"], "今天天气好")
        self.assertEqual(st, {"asked": 2, "switched": 1, "kept": 1, "unsure": 0})
        self.assertIn("超临界", seen[0])
        st2 = crosscheck.llm_arbitrate([seg(0, 0, "a天气", [{"primary": "天气", "secondary": "天汽", "pick": "primary"}])], [], lambda m: "not json")
        self.assertEqual(st2["unsure"], 1)


class NotesCheckTest(unittest.TestCase):
    def test_lines_and_missing(self):
        segs = [seg(0, 0, "六五零六的价格是一千四百五十美金"), seg(1, 65, "一个柜二十二点四吨，ME2500 有晶点问题"), seg(2, 130, "我们一个月用三百吨左右，175公斤试过")]
        notes = "6506 价格1450美元/吨\n单柜22.4吨，目标2柜\n客户月用约300吨\n6506 熔点134℃"
        r = crosscheck.notes_check(segs, notes, ["晶点", "ME2500"])
        st = {l["line"]: l["status"] for l in r["lines"]}
        self.assertEqual(st["6506 价格1450美元/吨"], "ok")
        self.assertEqual(st["客户月用约300吨"], "ok")
        self.assertEqual(st["6506 熔点134℃"], "partial")
        self.assertIn("134", [l for l in r["lines"] if l["line"].startswith("6506 熔点")][0]["missing"])
        keys = [e["key"] for e in r["missing_from_notes"]]
        self.assertIn("175", keys)
        self.assertIn("me2500", keys)
        md = crosscheck.report_md({"title": "t", "crosscheck": {"notes": r, "reference_vote": {"checked": 1, "switched": 0, "undecided": 1, "confirmed": 0}}})
        self.assertIn("회의록 줄별 확인", md)
        self.assertIn("175", md)


class CrosscheckJobTest(unittest.TestCase):
    def test_crosscheck_existing(self):
        tmp = Path(tempfile.mkdtemp(prefix="jl_cc_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "d").mkdir()
        (tmp / "d" / "acme.json").write_text(json.dumps({"terms": [{"term": "晶点"}], "replace": []}, ensure_ascii=False), encoding="utf-8")
        parent = {"title": "t", "customer": "acme", "warnings": [], "segments": [seg(0, 0, "有精点问题", [{"primary": "精点", "secondary": "晶点", "pick": "primary"}])]}
        ctx = pipeline.Ctx("cc1", tmp, {"parent_job_id": "p1", "references": [{"name": "doubao", "text": "有晶点问题"}], "notes": "晶点问题要解决", "arbitrate": True},
                           lambda s, f: None, lambda: False, lambda w: None)
        with mock.patch.object(config, "CUSTOMER_DICT_DIR", tmp / "d"), mock.patch.object(llm, "available", lambda: None):
            res = pipeline.crosscheck_existing(ctx, parent)
        self.assertEqual(res["segments"][0]["text"], "有晶点问题")
        self.assertEqual(res["crosscheck"]["customer"], "acme")
        self.assertEqual(res["crosscheck"]["notes"]["summary"]["ok"], 1)
        self.assertTrue(any("arbitration was skipped" in w for w in res["warnings"]))
        self.assertEqual(parent["segments"][0]["text"], "有精点问题")  # parent result is not modified


if __name__ == "__main__":
    unittest.main()
