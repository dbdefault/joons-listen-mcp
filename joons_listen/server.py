"""JOONS LISTEN MCP HTTP 서비스(127.0.0.1 전용). 표준 라이브러리만 사용."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from . import config, dictionary, engines, jobs, llm, store_sync

log = logging.getLogger("joons_listen.server")

LANGS = ["zh", "ko", "en", "ja"]
FORMATS = ("json", "txt", "md", "srt", "minutes", "crosscheck")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(Path(root).resolve())
        return True
    except Exception:  # noqa: BLE001
        return False


def find_by_name(name: str, max_depth: int = 3, max_entries: int = 20000) -> Optional[Path]:
    """허용 폴더(받은 파일 폴더 먼저)에서 이름이 같은 오디오 파일을 찾아 가장 최근 것을 돌려준다. 깊이·개수를 제한한다."""
    want = unicodedata.normalize("NFC", name).strip().lower()
    if not want or Path(want).suffix.lower() not in config.AUDIO_EXTS:
        return None
    seen = 0
    for root in config.ALLOWED_ROOTS:
        best: Optional[Path] = None
        best_m = -1.0
        stack = [(Path(root), 0)]
        while stack and seen < max_entries:
            d, depth = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        seen += 1
                        if seen >= max_entries:
                            break
                        try:
                            if e.is_dir(follow_symlinks=False):
                                if depth < max_depth:
                                    stack.append((Path(e.path), depth + 1))
                            elif unicodedata.normalize("NFC", e.name).lower() == want:
                                m = e.stat().st_mtime
                                if m > best_m:
                                    best, best_m = Path(e.path), m
                        except OSError:
                            continue
            except OSError:
                continue
        if best is not None:
            return best
    return None


def check_audio_path(raw) -> Path:
    if not raw or not isinstance(raw, str):
        raise ApiError(400, "BAD_REQUEST", "audio_path is required")
    p = Path(raw)
    if not p.is_absolute():
        if any(ch in raw for ch in ("/", "\\")):
            raise ApiError(400, "BAD_REQUEST", "audio_path must be an absolute path or a bare file name")
        found = find_by_name(raw)  # 파일 이름만 주면 허용 폴더에서 찾는다(ChatGPT·폰에서 경로 없이 부르기 쉽게)
        if found is None:
            raise ApiError(404, "FILE_NOT_FOUND", f"'{raw}' was not found in the allowed folders. Put the recording in an allowed folder (inbox, Desktop, Downloads or a folder set with JOONS_LISTEN_ROOTS) and try again.")
        p = found
    try:
        rp = p.resolve()
    except Exception:  # noqa: BLE001
        raise ApiError(400, "BAD_REQUEST", "the path cannot be resolved")
    if not any(_inside(rp, root) for root in config.ALLOWED_ROOTS):
        raise ApiError(403, "PATH_NOT_ALLOWED", "the file is outside the allowed folders (set them with JOONS_LISTEN_ROOTS)")
    low = str(rp).lower()
    if any(s in low for s in config.SECRET_NAME_PARTS):
        raise ApiError(403, "PATH_NOT_ALLOWED", "files whose names look like secrets are not processed")
    if not rp.is_file():
        raise ApiError(404, "FILE_NOT_FOUND", "file not found")
    if rp.suffix.lower() not in config.AUDIO_EXTS:
        raise ApiError(400, "BAD_REQUEST", "supported formats: " + ", ".join(sorted(config.AUDIO_EXTS)))
    return rp


class Service:
    def __init__(self) -> None:
        config.ensure_dirs()
        self.mgr = engines.EngineManager()
        self.jobs = jobs.JobManager(mgr=self.mgr, on_done=self._on_done)
        self.sync_lock = threading.Lock()
        self._sync_fail: Dict[str, tuple] = {}  # jid -> (연속 실패 횟수, 다음 시도 시각)
        self.started = time.time()

    def start(self) -> None:
        self.jobs.start()
        threading.Thread(target=self._sync_loop, daemon=True, name="listen-sync").start()

    # 결과 미러링: 끝난 직후 한 번, 실패하면 60초마다 다시(실패가 이어지면 간격을 늘려 최대 30분)
    def _on_done(self, jid: str, jdir: Path) -> None:
        threading.Thread(target=self._sync_one, args=(jid,), daemon=True).start()

    def _sync_one(self, jid: str) -> None:
        """예외가 나도 스레드가 죽지 않게 감싼다. 실패하면 pending 으로 남겨 다음 주기에 다시 한다."""
        try:
            with self.sync_lock:
                ok, msg = store_sync.push(jid, self.jobs.jobs_dir / jid, config.DATA_DIR)
                self.jobs.set_mirror(jid, "off" if msg == "off" else ("ok" if ok else "pending"))
                if ok or msg == "off":
                    self._sync_fail.pop(jid, None)
                else:
                    self._note_sync_fail(jid)
        except Exception as e:  # noqa: BLE001
            log.warning("mirror error %s: %s", jid, e)
            self._note_sync_fail(jid)
            try:
                self.jobs.set_mirror(jid, "pending")
            except Exception:  # noqa: BLE001
                pass

    def _note_sync_fail(self, jid: str) -> None:
        n = self._sync_fail.get(jid, (0, 0.0))[0] + 1
        self._sync_fail[jid] = (n, time.time() + min(1800, 60 * 2 ** min(n - 1, 5)))

    def _sync_loop(self) -> None:
        while not self.jobs.stop.wait(60):
            try:
                now = time.time()
                for jid in self.jobs.pending_sync():
                    if self._sync_fail.get(jid, (0, 0.0))[1] > now:
                        continue
                    self._sync_one(jid)
            except Exception as e:  # noqa: BLE001
                log.warning("mirror loop error: %s", e)

    # ---- 핸들러 ----
    def health(self) -> Dict:
        model = llm.available()
        pend = self.jobs.pending_sync()
        return {"ok": True, "version": config.VERSION, "gpu": engines.gpu_info(), "engines": self.mgr.status(), "llm": {"ready": True, "model": model} if model else None, "queue": self.jobs.counts(), "store": {"mirror": config.MIRROR_MODE, "pending": len(pend)}, "uptime_sec": int(time.time() - self.started)}

    def models(self) -> Dict:
        st = self.mgr.status()
        return {
            "ok": True,
            "engines": [
                {"id": "whisper", "name": self.mgr.whisper.name, "license": self.mgr.whisper.license, "languages": ["zh", "ko", "en", "ja", "+99"], "loaded": st["whisper"]["loaded"], "ready": st["whisper"]["ready"], "vram_mb": 3500, "notes": "int8_float16, 단어 힌트(hotwords) 지원"},
                {"id": "qwen", "name": self.mgr.qwen.name, "license": self.mgr.qwen.license, "languages": ["zh", "ko", "en", "ja", "+25"], "loaded": st["qwen"]["loaded"], "ready": st["qwen"]["ready"], "vram_mb": 5000, "notes": "중국어 방언·한국어에 강함(공식 발표), 자동 언어 인식"},
            ],
            "routing": {"zh": "qwen", "ko": "qwen", "en": "whisper", "mixed": "qwen"},
            "default_note": "초기 기본값이며 성능 우열이 확정된 것은 아닙니다. engine 으로 직접 고를 수 있습니다.",
        }

    def submit(self, body: Dict) -> Dict:
        kind = body.get("kind", "transcribe")
        if kind not in ("transcribe", "diarize"):
            raise ApiError(400, "BAD_REQUEST", "kind must be transcribe or diarize")
        rp = check_audio_path(body.get("audio_path"))
        lang = body.get("language", "auto") or "auto"
        if lang != "auto" and lang not in LANGS:
            raise ApiError(400, "BAD_REQUEST", "language must be auto|zh|ko|en|ja")
        eng = body.get("engine", "auto") or "auto"
        if eng not in ("auto", "whisper", "qwen", "ensemble"):
            raise ApiError(400, "BAD_REQUEST", "engine must be auto|whisper|qwen|ensemble")
        ns = body.get("num_speakers")
        if ns is not None and (not isinstance(ns, int) or isinstance(ns, bool) or ns < 1 or ns > 12):
            raise ApiError(400, "BAD_REQUEST", "num_speakers must be an integer from 1 to 12")
        cust = body.get("customer") or None
        if cust is not None:
            try:
                dictionary.customer_path(cust)
            except ValueError as e:
                raise ApiError(400, "BAD_REQUEST", str(e))
        params = {"audio_path": str(rp), "language": lang, "engine": eng, "diarize": bool(body.get("diarize", True)), "num_speakers": ns, "translate_ko": bool(body.get("translate_ko", False)), "minutes": bool(body.get("minutes", False)), "title": body.get("title"), "dictionary": body.get("dictionary"), "customer": cust}
        return {"ok": True, "job_id": self.jobs.submit(kind, params)}

    def summarize(self, jid: str, body: Dict) -> Dict:
        j = self.jobs.get(jid)
        if not j:
            raise ApiError(404, "NOT_FOUND", "job not found")
        if j["state"] != "done" or j["kind"] != "transcribe":
            raise ApiError(409, "NOT_READY", "only a finished transcription job can be summarized")
        params = {"parent_job_id": jid, "translate_ko": True, "minutes": True, "title": j.get("title")}
        return {"ok": True, "job_id": self.jobs.submit("summarize", params)}

    def crosscheck(self, jid: str, body: Dict) -> Dict:
        j = self.jobs.get(jid)
        if not j:
            raise ApiError(404, "NOT_FOUND", "job not found")
        if j["state"] != "done" or j["kind"] not in ("transcribe", "crosscheck"):
            raise ApiError(409, "NOT_READY", "only a finished transcription job can be cross-checked")
        notes = body.get("notes")
        if notes is not None and (not isinstance(notes, str) or len(notes) > 100000):
            raise ApiError(400, "BAD_REQUEST", "notes must be a string of at most 100000 characters")
        refs = body.get("references") or []
        if not isinstance(refs, list) or len(refs) > 5 or not all(isinstance(r, dict) and isinstance(r.get("text", ""), str) and len(r.get("text", "")) <= 100000 for r in refs):
            raise ApiError(400, "BAD_REQUEST", "references must be a list of at most 5 {name, text} objects (text up to 100000 characters; the whole request must stay under 1 MB)")
        cust = body.get("customer") or None
        if cust is not None:
            try:
                dictionary.customer_path(cust)
            except ValueError as e:
                raise ApiError(400, "BAD_REQUEST", str(e))
        if not notes and not refs and not body.get("arbitrate"):
            raise ApiError(400, "BAD_REQUEST", "give notes, references or arbitrate=true")
        params = {"parent_job_id": jid, "notes": notes, "references": [{"name": str(r.get("name") or f"ref{i + 1}")[:40], "text": r.get("text", "")} for i, r in enumerate(refs)],
                  "arbitrate": bool(body.get("arbitrate", False)), "customer": cust, "title": j.get("title")}
        return {"ok": True, "job_id": self.jobs.submit("crosscheck", params)}

    def status(self, jid: str) -> Dict:
        j = self.jobs.get(jid)
        if not j:
            raise ApiError(404, "NOT_FOUND", "job not found")
        return {"ok": True, **j}

    def result(self, jid: str, q: Dict) -> Dict:
        j = self.jobs.get(jid)
        if not j:
            raise ApiError(404, "NOT_FOUND", "job not found")
        if j["state"] != "done":
            raise ApiError(409, "NOT_READY", f"no result yet (state {j['state']})")
        fmt = q.get("format", ["md"])[0] or "md"
        if fmt not in FORMATS:
            raise ApiError(400, "BAD_REQUEST", "format must be one of " + "|".join(FORMATS))
        text = self.jobs.result_text(jid, fmt)
        if text is None:
            raise ApiError(404, "NOT_FOUND", "no result in that format")
        try:
            off = max(0, int(q.get("offset", ["0"])[0]))
            mx = max(1000, min(200000, int(q.get("max_chars", ["20000"])[0])))
        except ValueError:
            raise ApiError(400, "BAD_REQUEST", "offset and max_chars must be integers")
        chunk = text[off : off + mx]
        nxt = off + mx if off + mx < len(text) else None
        return {"ok": True, "job_id": jid, "format": fmt, "offset": off, "next_offset": nxt, "total_chars": len(text), "text": chunk}


def make_handler(svc: Service):
    class H(BaseHTTPRequestHandler):
        server_version = "JoonsListen/" + config.VERSION

        def log_message(self, *a):
            pass

        def _send(self, status: int, obj: Dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> Dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise ApiError(400, "BAD_REQUEST", "invalid Content-Length")
            if n < 0:  # 음수면 read(-1)이 연결이 끊길 때까지 기다리므로 먼저 거절한다
                raise ApiError(400, "BAD_REQUEST", "invalid Content-Length")
            if n > 1_000_000:
                raise ApiError(413, "BAD_REQUEST", "request body too large")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                d = json.loads(raw.decode("utf-8") or "{}")
            except Exception:  # noqa: BLE001
                raise ApiError(400, "BAD_REQUEST", "cannot parse the JSON body")
            if not isinstance(d, dict):
                raise ApiError(400, "BAD_REQUEST", "the body must be a JSON object")
            return d

        def _route(self, method: str) -> Tuple[int, Dict]:
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            q = parse_qs(u.query)
            if method == "GET":
                if parts == ["health"]:
                    return 200, svc.health()
                if parts == ["models"]:
                    return 200, svc.models()
                if parts == ["jobs"]:
                    return 200, {"ok": True, "jobs": svc.jobs.list(50)}
                if len(parts) == 2 and parts[0] == "jobs":
                    return 200, svc.status(parts[1])
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "result":
                    return 200, svc.result(parts[1], q)
            if method == "POST":
                if parts == ["jobs"]:
                    return 202, svc.submit(self._body())
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "summarize":
                    return 202, svc.summarize(parts[1], self._body())
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "crosscheck":
                    return 202, svc.crosscheck(parts[1], self._body())
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                    st = svc.jobs.cancel(parts[1])
                    if st is None:
                        raise ApiError(404, "NOT_FOUND", "job not found")
                    return 200, {"ok": True, "state": st}
            raise ApiError(404, "NOT_FOUND", "unknown route")

        def _guard(self, method: str) -> None:
            """로컬 웹 페이지가 몰래 부르는 것(교차 출처 요청, DNS 리바인딩)을 막는다."""
            host = (self.headers.get("Host") or "").strip().lower()
            port = self.server.server_address[1]
            ok_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}", "127.0.0.1", "localhost", "[::1]"}
            if host not in ok_hosts:
                raise ApiError(403, "BAD_REQUEST", "Host header not allowed")
            if self.headers.get("Origin"):
                raise ApiError(403, "BAD_REQUEST", "cross-origin browser requests are refused")
            if method == "POST" and "application/json" not in (self.headers.get("Content-Type") or "").lower():
                raise ApiError(415, "BAD_REQUEST", "Content-Type must be application/json")

        def _handle(self, method: str) -> None:
            try:
                self._guard(method)
                status, obj = self._route(method)
            except ApiError as e:
                status, obj = e.status, {"ok": False, "code": e.code, "message": e.message}
            except Exception as e:  # noqa: BLE001
                status, obj = 500, {"ok": False, "code": "INTERNAL", "message": str(e)[:200]}
            self._send(status, obj)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return H


def serve(host: str = config.HOST, port: int = config.PORT) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("security: the JOONS LISTEN service binds to loopback only")
    # 포트를 먼저 묶어 서비스가 둘 뜨는 것을 막는다(둘째는 여기서 실패하고, 작업 DB 는 건드리지 않는다)
    holder: Dict = {}

    class _Lazy:
        def __getattr__(self, k):
            return getattr(holder["svc"], k)

    ThreadingHTTPServer.allow_reuse_address = False
    httpd = ThreadingHTTPServer((host, port), make_handler(_Lazy()))
    svc = Service()
    holder["svc"] = svc
    svc.start()
    print(f"JOONS LISTEN {config.VERSION} http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        svc.jobs.shutdown()
