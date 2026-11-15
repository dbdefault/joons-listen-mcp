import shutil
import tempfile
import unittest
from pathlib import Path

from joons_listen import dictionary


class DictionaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jl_dict_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = {
            "terms": [
                {"term": "Acme(艾克米)", "type": "company", "aliases": ["艾克", "에이크미", "Acme"]},
                {"term": "Globex(格罗贝)", "aliases": ["格罗贝"]},
            ],
            "replace": [{"from": "Akme", "to": "Acme"}],
        }

    def test_save_load_roundtrip(self):
        p = self.tmp / "sub" / "d.json"
        dictionary.save(self.data, p)
        self.assertEqual(dictionary.load(p), self.data)
        self.assertEqual(list(p.parent.glob("*.tmp")), [])

    def test_load_missing_and_broken(self):
        self.assertEqual(dictionary.load(self.tmp / "no.json"), {"terms": [], "replace": []})
        bad = self.tmp / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        self.assertEqual(dictionary.load(bad), {"terms": [], "replace": []})

    def test_load_bom_and_defaults(self):
        p = self.tmp / "b.json"
        p.write_text('{"terms": []}', encoding="utf-8-sig")
        self.assertEqual(dictionary.load(p), {"terms": [], "replace": []})

    def test_hint_words_dedupe_and_limit(self):
        w = dictionary.hint_words(self.data)
        self.assertEqual(w, ["Acme(艾克米)", "艾克", "에이크미", "Acme", "Globex(格罗贝)", "格罗贝"])
        self.assertEqual(len(dictionary.hint_words(self.data, limit=2)), 2)
        self.assertEqual(dictionary.hint_words({}), [])

    def test_merge_job_replace(self):
        r = dictionary.merge_job_replace(self.data, [{"from": "A", "to": "B"}, {"from": ""}, "x", {"from": "C"}])
        self.assertEqual(r, [{"from": "Akme", "to": "Acme"}, {"from": "A", "to": "B"}, {"from": "C", "to": ""}])
        self.assertEqual(len(self.data["replace"]), 1)
        self.assertEqual(dictionary.merge_job_replace(self.data, None), self.data["replace"])

    def test_apply_replace(self):
        rules = [{"from": "akme", "to": "Acme"}, {"from": "a.b", "to": "X"}, {"from": ""}]
        self.assertEqual(dictionary.apply_replace("Akme 와 a.b 그리고 aXb", rules), "Acme 와 X 그리고 aXb")

    def test_name_hits(self):
        self.assertEqual(dictionary.name_hits("艾克那边说可以", self.data), ["Acme(艾克米)"])
        self.assertEqual(dictionary.name_hits("ACME and 格罗贝", self.data), ["Acme(艾克米)", "Globex(格罗贝)"])
        self.assertEqual(dictionary.name_hits("아무 얘기 없음", self.data), [])

    def test_build_from_folder_names(self):
        d = dictionary.build_from_folder_names(["01_Acme艾克米", "03_Globex格罗贝", "readme", "02_Acme艾克"])
        self.assertEqual([t["term"] for t in d["terms"]], ["Acme(艾克米)", "Globex(格罗贝)"])


if __name__ == "__main__":
    unittest.main()
