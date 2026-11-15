"""독립 검토에서 나온 문제의 회귀 시험."""
import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from joons_listen import dictionary, llm, server, store_sync


class DictionaryBackslash(unittest.TestCase):
    def test_to_with_backslash_is_literal(self):
        out = dictionary.apply_replace("path a here", [{"from": "a", "to": "C:\\x\\1"}])
        self.assertIn("C:\\x\\1", out)


class LocalOnlyLlm(unittest.TestCase):
    def test_is_local(self):
        self.assertTrue(llm._is_local("http://127.0.0.1:11434"))
        self.assertTrue(llm._is_local("http://localhost:11434/api"))
        self.assertFalse(llm._is_local("http://example.com:11434"))
        self.assertFalse(llm._is_local("http://127.0.0.1.evil.com:11434"))


class SyncCleanup(unittest.TestCase):
    def test_push_failure_keeps_pending_and_bundle_removed_only_on_success(self):
        t = Path(tempfile.mkdtemp())
        try:
            j = t / "job"
            j.mkdir()
            (j / "result.json").write_text("{}")
            from joons_listen import config

            blocker = t / "file_not_dir"
            blocker.write_text("x")
            old = (config.MIRROR_MODE, config.MIRROR_DIR)
            config.MIRROR_MODE, config.MIRROR_DIR = "folder", str(blocker / "sub")  # 폴더를 만들 수 없는 경로
            try:
                ok, msg = store_sync.push("jx", j, t)
            finally:
                config.MIRROR_MODE, config.MIRROR_DIR = old
            self.assertFalse(ok)
            self.assertTrue((t / "sync" / "jx.zip").exists())
        finally:
            shutil.rmtree(t, ignore_errors=True)


class HttpGuard(unittest.TestCase):
    """Host·Origin·Content-Type 검사와 라우트(작업 관리자는 가짜)."""

    @classmethod
    def setUpClass(cls):
        cls.t = Path(tempfile.mkdtemp())
        from joons_listen import config, jobs

        cls._old = (config.DATA_DIR, config.DB_PATH, config.JOBS_DIR)

        class FakeSvc:
            def __init__(self):
                self.jobs = jobs.JobManager(db_path=cls.t / "j.sqlite3", jobs_dir=cls.t / "jobs", runner=lambda ctx, kind: {"segments": [], "speakers": [], "source": {}, "warnings": []})

            def health(self):
                return {"ok": True}

            status = server.Service.status
            result = server.Service.result
            summarize = server.Service.summarize
            submit = server.Service.submit

        cls.svc = FakeSvc()
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(cls.svc))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.svc.jobs.shutdown()
        shutil.rmtree(cls.t, ignore_errors=True)

    def _req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = json.loads(r.read().decode("utf-8"))
        c.close()
        return r.status, data

    def test_health_ok(self):
        self.assertEqual(self._req("GET", "/health")[0], 200)

    def test_foreign_host_rejected(self):
        self.assertEqual(self._req("GET", "/health", headers={"Host": "evil.example.com"})[0], 403)

    def test_ipv6_loopback_host_allowed(self):
        self.assertEqual(self._req("GET", "/health", headers={"Host": f"[::1]:{self.port}"})[0], 200)

    def test_origin_rejected(self):
        self.assertEqual(self._req("GET", "/health", headers={"Origin": "http://evil.example.com"})[0], 403)

    def test_text_plain_post_rejected(self):
        self.assertEqual(self._req("POST", "/jobs", body="{}", headers={"Content-Type": "text/plain"})[0], 415)

    def test_unknown_route_404(self):
        self.assertEqual(self._req("GET", "/nope")[0], 404)

    def test_unknown_job_404_and_not_ready_409(self):
        self.assertEqual(self._req("GET", "/jobs/lst_nope")[0], 404)



class ChineseFactsAndNameLookup(unittest.TestCase):
    def test_chinese_numeral_price_and_qty(self):
        from joons_listen import verify

        f = {(x["type"], x["text"]) for x in verify.extract_facts("我们成本他要两万一吨，国内才买一万二一升", 0)}
        self.assertIn(("price", "两万一吨"), f)
        self.assertNotIn(("quantity", "两万一吨"), f)
        g = {(x["type"], x["text"]) for x in verify.extract_facts("挤出量400kg-500kg/h，日需求1.5-1.8mt/吨，目前8条生产线", 0)}
        self.assertIn(("quantity", "8条"), g)
        self.assertTrue(any(t == "quantity" and "1.5-1.8mt" in x for t, x in g))
        h = {(x["type"], x["text"]) for x in verify.extract_facts("价格是1100美元一吨，下个月十五号交货", 0)}
        self.assertIn(("price", "1100美元"), h)
        self.assertIn(("date", "十五号"), h)

    def test_find_by_name(self):
        from joons_listen import config

        t = Path(tempfile.mkdtemp())
        try:
            (t / "sub").mkdir()
            f = t / "sub" / "회의 녹음.m4a"
            f.write_bytes(b"x")
            old = config.ALLOWED_ROOTS
            config.ALLOWED_ROOTS = [t]
            try:
                self.assertEqual(server.find_by_name("회의 녹음.m4a"), f)
                self.assertIsNone(server.find_by_name("없는파일.m4a"))
                self.assertIsNone(server.find_by_name("notes.txt"))  # 오디오 확장자가 아니면 찾지 않는다
                self.assertEqual(server.check_audio_path("회의 녹음.m4a"), f.resolve())
            finally:
                config.ALLOWED_ROOTS = old
        finally:
            shutil.rmtree(t, ignore_errors=True)


class CnyAndTrivial(unittest.TestCase):
    def test_fix_cny(self):
        self.assertEqual(llm.fix_cny("월 200~300원에 사무실, 연 매출 1억 원 이상, 2만 원/톤", "zh"), "월 200~300위안에 사무실, 연 매출 1억 위안 이상, 2만 위안/톤")
        self.assertEqual(llm.fix_cny("원료를 구매", "zh"), "원료를 구매")  # '원' 뒤에 한글이 이어지면 건드리지 않는다
        self.assertEqual(llm.fix_cny("200원", "ko"), "200원")  # 한국어 원문은 그대로

    def test_trivial_counts_dropped(self):
        from joons_listen import verify

        f = {(x["type"], x["text"]) for x in verify.extract_facts("买一个，一块，一套，五十个，两三百块钱", 0)}
        self.assertNotIn(("quantity", "一个"), f)
        self.assertNotIn(("price", "一块"), f)
        self.assertIn(("quantity", "五十个"), f)


class ZWorkerFixes(unittest.TestCase):
    def test_negative_content_length_rejected_fast(self):
        import socket

        from joons_listen import jobs

        t = Path(tempfile.mkdtemp())
        try:
            class Fake:
                submit = server.Service.submit  # 본문을 읽기 전에 속성부터 찾으므로 있어야 한다

                def __init__(self):
                    self.jobs = jobs.JobManager(db_path=t / "j.sqlite3", jobs_dir=t / "jobs", runner=lambda ctx, kind: {"segments": [], "speakers": [], "source": {}, "warnings": []})

            httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(Fake()))
            port = httpd.server_address[1]
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.sendall(f"POST /jobs HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: -1\r\n\r\n".encode())
            data = s.recv(4096).decode("utf-8", "replace")
            s.close()
            httpd.shutdown()
            self.assertIn("400", data.split("\r\n", 1)[0])
        finally:
            shutil.rmtree(t, ignore_errors=True)

    def test_cn_value_units(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("eval_fleurs", Path(__file__).resolve().parents[1] / "tools" / "eval_fleurs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertEqual(m._cn_to_int("一万亿"), "1000000000000")
        self.assertEqual(m._cn_to_int("三亿五千万"), "350000000")
        self.assertEqual(m._cn_to_int("一万九千五百"), "19500")
        self.assertEqual(m._cn_to_int("十四"), "14")
        self.assertEqual(m._cn_to_int("二零二四"), "2024")

    def test_empty_checklist_item_is_error(self):
        import subprocess
        import sys

        d = Path(tempfile.mkdtemp())
        try:
            (d / "c.json").write_text(json.dumps([{"item": "빈 항목"}], ensure_ascii=False), encoding="utf-8")
            (d / "t.txt").write_text("아무 글", encoding="utf-8")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "tools" / "eval_recording.py"), "--checklist", str(d / "c.json"), "--text", str(d / "t.txt")], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class SyncResilience(unittest.TestCase):
    """동기화가 예외를 만나도 스레드가 죽지 않고, 실패하면 간격을 늘린다."""

    def _svc(self):
        svc = server.Service.__new__(server.Service)
        svc.sync_lock = threading.Lock()
        svc._sync_fail = {}
        calls = []

        class J:
            jobs_dir = Path(tempfile.gettempdir())

            def set_mirror(self, jid, st):
                calls.append((jid, st))

        svc.jobs = J()
        return svc, calls

    def test_exception_keeps_pending_and_backs_off(self):
        from joons_listen import store_sync

        svc, calls = self._svc()
        old = store_sync.push
        store_sync.push = lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked"))
        try:
            svc._sync_one("lst_x")
        finally:
            store_sync.push = old
        self.assertEqual(calls[-1], ("lst_x", "pending"))
        n, nxt = svc._sync_fail["lst_x"]
        self.assertEqual(n, 1)
        self.assertGreater(nxt, time.time())

    def test_success_clears_failure(self):
        from joons_listen import store_sync

        svc, calls = self._svc()
        svc._sync_fail["lst_y"] = (3, time.time() + 999)
        old = store_sync.push
        store_sync.push = lambda *a, **k: (True, "ok")
        try:
            svc._sync_one("lst_y")
        finally:
            store_sync.push = old
        self.assertNotIn("lst_y", svc._sync_fail)
        self.assertEqual(calls[-1], ("lst_y", "ok"))

    def test_backoff_capped(self):
        svc, _ = self._svc()
        for _ in range(12):
            svc._note_sync_fail("lst_z")
        n, nxt = svc._sync_fail["lst_z"]
        self.assertLessEqual(nxt - time.time(), 1800 + 1)



class VadFillGaps(unittest.TestCase):
    def test_long_gap_filled_and_silence_skipped(self):
        import numpy as np
        from joons_listen import vad

        sr = 16000
        a = (np.random.RandomState(0).randn(sr * 100) * 0.05).astype("float32")
        a[sr * 60 :] = 0  # 60초 이후는 완전 무음
        out = vad.fill_gaps([(1.0, 3.0), (90.0, 95.0)], a, sr)
        self.assertTrue(any(30 < s < 60 for s, e in out))  # 소리 있는 빈 구간은 채움
        self.assertFalse(any(65 < s < 85 for s, e in out))  # 무음 구간은 안 채움
        self.assertIn((1.0, 3.0), out)
        self.assertIn((90.0, 95.0), out)
        self.assertTrue(all(e - s <= 25.01 for s, e in out if (s, e) not in [(1.0, 3.0), (90.0, 95.0)]))

    def test_short_gap_left_alone(self):
        import numpy as np
        from joons_listen import vad

        a = np.ones(16000 * 20, dtype="float32") * 0.1
        self.assertEqual(vad.fill_gaps([(0.0, 8.0), (15.0, 19.0)], a, 16000), [(0.0, 8.0), (15.0, 19.0)])

    def test_empty_vad_falls_back_to_chunks(self):
        import numpy as np
        from joons_listen import vad

        a = np.ones(16000 * 60, dtype="float32") * 0.1
        out = vad.fill_gaps([], a, 16000)
        self.assertGreaterEqual(len(out), 2)



class HintEcho(unittest.TestCase):
    H = ["Acme", "艾克美", "Globex", "环球", "Initech", "英特科", "Umbrella", "雨伞", "Hooli"]

    def test_echo_detected(self):
        from joons_listen import dictionary

        t = "Acme(艾克美), Acme, 艾克美, Globex(环球), Globex, 环球, Initech(英特科), Umbrella, 雨伞, Hooli"
        self.assertTrue(dictionary.is_hint_echo(t, self.H))

    def test_real_speech_with_names_kept(self):
        from joons_listen import dictionary

        t = "我们跟雨伞合作很多年了，Hooli那边最近也在测试新的材料，而且价格方面双方都比较满意，下个月还要继续谈。"
        self.assertFalse(dictionary.is_hint_echo(t, self.H))

    def test_short_or_empty(self):
        from joons_listen import dictionary

        self.assertFalse(dictionary.is_hint_echo("", self.H))
        self.assertFalse(dictionary.is_hint_echo("雨伞", self.H))


if __name__ == "__main__":
    unittest.main()



class ZWorkerRegexFixes(unittest.TestCase):
    def facts(self, text):
        from joons_listen import verify

        return {(x["type"], x["text"]) for x in verify.extract_facts(text, 0)}

    def test_codes_adjacent_to_cjk(self):
        self.assertIn(("code", "PP-1100N"), self.facts("型号PP-1100N，数量500吨"))
        self.assertIn(("code", "PP-1100N"), self.facts("모델은 PP-1100N입니다"))
        self.assertIn(("code", "SO-20261115"), self.facts("주문번호 SO-20261115 접수"))

    def test_english_month_year(self):
        self.assertIn(("date", "March 2026"), self.facts("Contract starts March 2026"))
        self.assertIn(("date", "20 Nov 2026"), self.facts("Delivery 20 Nov 2026"))

    def test_multipliers_and_prefixed_currency(self):
        self.assertIn(("quantity", "5万吨"), self.facts("这个月发5万吨"))
        self.assertIn(("quantity", "2만톤"), self.facts("이번 물량은 2만톤"))
        self.assertIn(("price", "3억원"), self.facts("계약 3억원"))
        f = self.facts("USD 850 and RMB 500, 人民币1200")
        self.assertTrue({("price", "USD 850"), ("price", "RMB 500"), ("price", "人民币1200")} <= f)

    def test_false_positives_removed(self):
        self.assertFalse([x for x in self.facts("공사는 3개월 걸립니다") if x[0] == "quantity"])
        self.assertFalse([x for x in self.facts("三个月以内交货") if x[0] == "quantity"])
        self.assertFalse([x for x in self.facts("수율은 95/100입니다 we split 50/50") if x[0] == "date"])

    def test_summarize_path_applies_cny_fix(self):
        from joons_listen import llm, pipeline

        old = (llm.available, llm.translate_lines, llm.make_minutes, llm.unload)
        llm.available = lambda: "m"
        llm.translate_lines = lambda lines, model, batch=12: ["단가 5000원" for _ in lines]
        llm.make_minutes = lambda *a, **k: {"title": "t", "summary": "", "decisions": [], "actions": [], "open_questions": [], "llm": "m"}
        llm.unload = lambda *a, **k: None
        try:
            ctx = pipeline.Ctx("j", Path(tempfile.mkdtemp()), {"translate_ko": True}, lambda s, f: None, lambda: False, lambda m: None)
            parent = {"segments": [{"i": 0, "text": "单价5000元", "lang": "zh", "speaker": "S1", "start": 0, "end": 1}], "speakers": [{"id": "S1", "label": "화자 1"}], "facts": [], "warnings": []}
            res = pipeline.summarize_existing(ctx, parent)
            self.assertEqual(res["segments"][0]["text_ko"], "단가 5000위안")
        finally:
            llm.available, llm.translate_lines, llm.make_minutes, llm.unload = old
            shutil.rmtree(ctx.dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
