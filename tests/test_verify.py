import unittest

from joons_listen import verify


def texts(facts, kind):
    return [f["text"] for f in facts if f["type"] == kind]


class ExtractFactsTest(unittest.TestCase):
    def test_chinese(self):
        f = verify.extract_facts("这批货单价 1,200 元/吨,数量 50吨,3月15日交货,型号 PP-1100N", 3)
        self.assertTrue(any("1,200" in t for t in texts(f, "price")))
        self.assertTrue(any(t.startswith("50") for t in texts(f, "quantity")))
        self.assertTrue(any("3月15日" in t for t in texts(f, "date")))
        self.assertTrue(any("PP" in t for t in texts(f, "code")))
        self.assertTrue(all(x["segment"] == 3 and x["verify"] is True for x in f))

    def test_korean(self):
        f = verify.extract_facts("단가는 850 달러, 수량은 20톤이고 12월 5일까지 납품, 다음 주 월요일 확인", 0)
        self.assertTrue(any("850" in t for t in texts(f, "price")))
        self.assertTrue(any("20" in t for t in texts(f, "quantity")))
        self.assertTrue(any("12월 5일" in t for t in texts(f, "date")))

    def test_english(self):
        f = verify.extract_facts("Price is $1,250 per ton, 500 kg sample, ship by Nov 20th 2026, grade ABS750", 1)
        self.assertTrue(any("$1,250" in t for t in texts(f, "price")))
        self.assertTrue(any("500" in t for t in texts(f, "quantity")))
        self.assertTrue(any("Nov 20" in t for t in texts(f, "date")))
        self.assertIn("ABS750", texts(f, "code"))

    def test_no_facts(self):
        self.assertEqual(verify.extract_facts("안녕하세요 반갑습니다", 0), [])

    def test_dedupe(self):
        f = verify.extract_facts("30吨 和 30吨", 0)
        self.assertEqual(len(texts(f, "quantity")), 1)

    def test_stop_codes_not_code(self):
        f = verify.extract_facts("MT KG RMB USD EA", 0)
        self.assertEqual(texts(f, "code"), [])

    def test_currency_prefix_not_code(self):
        # 알려진 오탐: "USD 100" 이 가격이 아니라 제품 코드로 잡힌다(보고용, verify.py 는 수정하지 않음).
        f = verify.extract_facts("USD 100", 0)
        self.assertEqual(texts(f, "code"), [])


class FlagSegmentTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(verify.flag_segment("   ", 3.0, 0.9), ["empty"])

    def test_low_confidence(self):
        self.assertEqual(verify.flag_segment("정상적인 문장입니다 그렇습니다", 3.0, 0.3), ["uncertain"])

    def test_normal_no_flags(self):
        self.assertEqual(verify.flag_segment("오늘 회의를 시작하겠습니다", 3.0, 0.9), [])

    def test_confidence_none_ok(self):
        self.assertEqual(verify.flag_segment("오늘 회의를 시작하겠습니다", 3.0, None), [])

    def test_too_fast_or_slow(self):
        self.assertIn("uncertain", verify.flag_segment("가" * 60, 3.0, 0.9))
        self.assertIn("uncertain", verify.flag_segment("네", 5.0, 0.9))

    def test_short_duration_skips_speed_check(self):
        self.assertEqual(verify.flag_segment("네", 1.0, 0.9), [])

    def test_repeat_hallucination(self):
        fl = verify.flag_segment("谢谢谢谢谢谢谢谢谢谢谢谢谢谢", 5.0, 0.9)
        self.assertEqual(fl, ["repeat", "uncertain"])


class AnalyzeTest(unittest.TestCase):
    def test_analyze(self):
        segs = [
            {"i": 0, "text": "단가 800 달러입니다", "start": 0, "end": 3, "confidence": 0.5},
            {"i": 1, "text": "좋습니다 알겠습니다 감사합니다", "start": 3, "end": 6, "confidence": 0.95},
            {"i": 2, "text": "", "start": 6, "end": 7, "confidence": 0.9},
            {"i": 3, "text": "수량은 10톤입니다", "start": 7, "end": 10, "confidence": 0.9},
        ]
        r = verify.analyze(segs)
        self.assertIn("price", segs[0]["flags"])
        self.assertIn("check_price", segs[0]["flags"])
        self.assertEqual(segs[1]["flags"], [])
        self.assertEqual(segs[2]["flags"], ["empty"])
        self.assertIn("quantity", segs[3]["flags"])
        self.assertNotIn("check_quantity", segs[3]["flags"])
        self.assertEqual(r["uncertain"], [0])
        self.assertTrue(all(f["verify"] for f in r["facts"]))

    def test_analyze_empty(self):
        self.assertEqual(verify.analyze([]), {"facts": [], "uncertain": []})


if __name__ == "__main__":
    unittest.main()
