import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from joons_listen import config, dictionary, ensemble


class DigitsTest(unittest.TestCase):
    def test_digit_by_digit_readings(self):
        self.assertEqual(ensemble.digits_to_arabic("六五零六的价格是一四五零美金"), "6506的价格是1450美金")
        self.assertEqual(ensemble.digits_to_arabic("幺幺零零"), "1100")

    def test_short_or_unit_numbers_untouched(self):
        self.assertEqual(ensemble.digits_to_arabic("一个柜二十二点四吨"), "一个柜二十二点四吨")
        self.assertEqual(ensemble.digits_to_arabic("两三个"), "两三个")


class MergeTest(unittest.TestCase):
    TERMS = ["超临界", "陶氏", "Acme"]

    def test_identical(self):
        m = ensemble.merge("我们做超临界发泡。", "我们做超临界发泡", self.TERMS)
        self.assertEqual(m["agree"], 1.0)
        self.assertEqual(m["conflicts"], [])
        self.assertEqual(m["text"], "我们做超临界发泡。")  # primary punctuation kept

    def test_secondary_wins_with_dictionary_term(self):
        m = ensemble.merge("我们做超链接发泡，用桃树的料。", "我们做超临界发泡用陶氏的料", self.TERMS)
        self.assertIn("超临界", m["text"])
        self.assertIn("陶氏", m["text"])
        self.assertIn("，", m["text"])
        self.assertTrue(all(c["pick"] == "secondary" for c in m["conflicts"]))

    def test_primary_kept_without_evidence(self):
        m = ensemble.merge("今天天气很好", "今天天汽很好", self.TERMS)
        self.assertEqual(m["text"], "今天天气很好")
        self.assertEqual(m["conflicts"][0]["pick"], "primary")
        self.assertLess(m["agree"], 1.0)

    def test_numbers_normalised_before_compare(self):
        m = ensemble.merge("价格一四五零美金", "价格1450美金", self.TERMS)
        self.assertEqual(m["agree"], 1.0)
        self.assertEqual(m["text"], "价格1450美金")

    def test_unit_numbers_count_as_agreement(self):
        m = ensemble.merge("一个柜二十吨，一千六百美金", "一个柜20吨1600美金", self.TERMS)
        self.assertEqual(m["agree"], 1.0)
        self.assertEqual(m["conflicts"], [])
        self.assertEqual(m["text"], "一个柜二十吨，一千六百美金")

    def test_arabic_digits_alone_do_not_win(self):
        # Field case: Whisper wrote a different number (1.7 for 零点七).
        m = ensemble.merge("现在就是一点零嘛零点七零点八啊", "现在就是1.0嘛1.7 1.8啊", self.TERMS)
        self.assertIn("零点七", m["text"])
        self.assertTrue(all(c["pick"] == "primary" for c in m["conflicts"]))

    def test_short_term_does_not_replace_long_span(self):
        # Field case: a whole clause was replaced by one dictionary word.
        terms = self.TERMS + ["TPU"]
        m = ensemble.merge("因为它要从原材料那边就要去一个颜色的附着", "因为它要从TPU颜色的附着", terms)
        self.assertIn("原材料", m["text"])
        self.assertNotIn("TPU", m["text"])

    def test_term_in_context_only_does_not_win(self):
        terms = ["陶氏"]
        m = ensemble.merge("陶氏的料今天天气很好", "陶氏的料今天天汽很好", terms)
        self.assertEqual(m["text"], "陶氏的料今天天气很好")

    def test_secondary_only_insertion_ignored(self):
        m = ensemble.merge("好的", "好的谢谢大家再见", self.TERMS)
        self.assertEqual(m["text"], "好的")
        self.assertEqual(m["conflicts"][0]["pick"], "primary")

    def test_empty_sides(self):
        self.assertEqual(ensemble.merge("", "", self.TERMS)["text"], "")
        self.assertEqual(ensemble.merge("你好", "", self.TERMS)["text"], "你好")
        self.assertEqual(ensemble.merge("", "你好", self.TERMS)["text"], "")

    def test_korean_spacing_kept(self):
        m = ensemble.merge("에이크미 제품을 씁니다", "Acme 제품을 씁니다", self.TERMS)
        self.assertEqual(m["text"], "Acme 제품을 씁니다")


class CustomerDictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_cust_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_load_for_merges_customer_first(self):
        g = self.tmp / "global.json"
        dictionary.save({"terms": [{"term": "G"}], "replace": [{"from": "x", "to": "y"}]}, g)
        with mock.patch.object(config, "DICT_PATH", g), mock.patch.object(config, "CUSTOMER_DICT_DIR", self.tmp / "d"):
            dictionary.save({"terms": [{"term": "C"}], "replace": [{"from": "桃树", "to": "陶氏"}]}, dictionary.customer_path("anta"))
            d = dictionary.load_for("anta")
            self.assertEqual([t["term"] for t in d["terms"]], ["C", "G"])
            self.assertEqual(d["replace"][0]["to"], "陶氏")
            self.assertEqual(dictionary.load_for(None)["terms"], [{"term": "G"}])
            self.assertEqual(dictionary.load_for("missing")["terms"], [{"term": "G"}])

    def test_customer_name_rejects_paths(self):
        for bad in ("../x", "a/b", "a\\b", "", "-x", "x" * 65):
            with self.assertRaises(ValueError):
                dictionary.customer_path(bad)


if __name__ == "__main__":
    unittest.main()
