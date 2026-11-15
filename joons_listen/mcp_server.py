"""MCP 서버(표준 입출력, 줄 단위 JSON-RPC 2.0). 표준 라이브러리만 사용한다.
MCP 클라이언트(Claude Desktop, Claude Code 등)가 이 프로그램을 자식 프로세스로 띄우면
실행 중인 JOONS LISTEN HTTP 서비스(`python -m joons_listen serve`)를 도구로 부를 수 있다.
서비스 주소는 JOONS_LISTEN_URL(기본 http://127.0.0.1:18796). 루프백 주소만 허용한다."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from . import config

PROTOCOL = "2025-06-18"
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")
BASE_URL = os.environ.get("JOONS_LISTEN_URL", f"http://127.0.0.1:{config.PORT}").rstrip("/")


def _is_loopback(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def call_api(method: str, path: str, body: Optional[Dict] = None, timeout: int = 30) -> Dict:
    if not _is_loopback(BASE_URL):
        raise RuntimeError("JOONS_LISTEN_URL 은 루프백 주소(127.0.0.1)여야 합니다")
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 환경 프록시를 타지 않는다
    try:
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {"ok": False, "code": "HTTP_ERROR", "message": f"HTTP {e.code}"}
    except (urllib.error.URLError, OSError):
        return {"ok": False, "code": "SERVICE_DOWN", "message": f"JOONS LISTEN 서비스에 연결하지 못했습니다({BASE_URL}). `python -m joons_listen serve` 로 먼저 시작하세요."}


def _pick(args: Dict, keys: List[str]) -> Dict:
    return {k: args[k] for k in keys if k in args and args[k] is not None}


def _t_transcribe(a: Dict) -> Dict:
    return call_api("POST", "/jobs", {"kind": "transcribe", **_pick(a, ["audio_path", "language", "engine", "diarize", "num_speakers", "translate_ko", "minutes", "title", "dictionary"])})


def _t_diarize(a: Dict) -> Dict:
    return call_api("POST", "/jobs", {"kind": "diarize", **_pick(a, ["audio_path", "num_speakers", "title"])})


def _t_summarize(a: Dict) -> Dict:
    jid = urllib.parse.quote(str(a.get("job_id", "")), safe="")
    return call_api("POST", f"/jobs/{jid}/summarize", _pick(a, ["style", "force"]))


def _t_status(a: Dict) -> Dict:
    if a.get("job_id"):
        return call_api("GET", "/jobs/" + urllib.parse.quote(str(a["job_id"]), safe=""))
    h = call_api("GET", "/health")
    h["jobs"] = call_api("GET", "/jobs").get("jobs", [])
    return h


def _t_result(a: Dict) -> Dict:
    q = urllib.parse.urlencode({k: a[k] for k in ("format", "offset", "max_chars") if k in a and a[k] is not None})
    return call_api("GET", "/jobs/" + urllib.parse.quote(str(a.get("job_id", "")), safe="") + "/result" + (("?" + q) if q else ""))


def _t_models(a: Dict) -> Dict:
    return call_api("GET", "/models")


S = {"type": "string"}
B = {"type": "boolean"}
N = {"type": "number"}
TOOLS: List[Dict[str, Any]] = [
    {"name": "listen_transcribe", "description": "Queue a transcription job for a local audio file (Whisper large-v3 / Qwen3-ASR-1.7B, speaker labels, optional Korean translation and minutes). Returns a job_id immediately; poll listen_status, then fetch listen_result.", "fn": _t_transcribe, "schema": {"type": "object", "properties": {"audio_path": {**S, "description": "Path or bare file name inside an allowed folder"}, "language": {**S, "description": "auto|zh|ko|en"}, "engine": {**S, "description": "auto|whisper|qwen"}, "diarize": B, "num_speakers": {**N, "description": "1-10"}, "translate_ko": B, "minutes": B, "title": S, "dictionary": {"type": "array", "items": {"type": "object", "properties": {"from": S, "to": S}, "required": ["from", "to"]}}}, "required": ["audio_path"]}, "ro": False},
    {"name": "listen_diarize", "description": "Queue a speaker-turn job (who spoke when, no transcript).", "fn": _t_diarize, "schema": {"type": "object", "properties": {"audio_path": S, "num_speakers": N, "title": S}, "required": ["audio_path"]}, "ro": False},
    {"name": "listen_summarize", "description": "Create minutes or a brief summary from a finished transcription job.", "fn": _t_summarize, "schema": {"type": "object", "properties": {"job_id": S, "style": {**S, "description": "minutes|brief"}, "force": B}, "required": ["job_id"]}, "ro": False},
    {"name": "listen_status", "description": "Status of one job, or service health plus recent jobs when job_id is omitted.", "fn": _t_status, "schema": {"type": "object", "properties": {"job_id": S}}, "ro": True},
    {"name": "listen_result", "description": "Read a finished job's result in chunks (format json|txt|md|srt|minutes).", "fn": _t_result, "schema": {"type": "object", "properties": {"job_id": S, "format": S, "offset": N, "max_chars": N}, "required": ["job_id"]}, "ro": True},
    {"name": "listen_models", "description": "List the engines, their state and the language routing.", "fn": _t_models, "schema": {"type": "object", "properties": {}}, "ro": True},
]
_BY_NAME = {t["name"]: t for t in TOOLS}


def tools_list() -> List[Dict]:
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["schema"], "annotations": {"readOnlyHint": t["ro"], "openWorldHint": False}} for t in TOOLS]


def tool_call(name: str, args: Dict) -> Dict:
    t = _BY_NAME.get(name)
    if t is None:
        return {"isError": True, "content": [{"type": "text", "text": f"알 수 없는 도구입니다: {name}"}]}
    try:
        res = t["fn"](args or {})
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "code": "ERROR", "message": str(e)[:300]}
    bad = isinstance(res, dict) and res.get("ok") is False
    return {"isError": bad, "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False, indent=2)}]}


def handle(msg: Dict) -> Optional[Dict]:
    """JSON-RPC 요청 하나를 처리한다. 알림(id 없음)과 응답 메시지에는 None."""
    mid, method = msg.get("id"), msg.get("method")
    if method is None or mid is None:
        return None
    try:
        if method == "initialize":
            want = (msg.get("params") or {}).get("protocolVersion")
            res = {"protocolVersion": want if want in SUPPORTED else PROTOCOL, "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "joons-listen-mcp", "version": config.VERSION}}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": tools_list()}
        elif method == "tools/call":
            p = msg.get("params") or {}
            res = tool_call(p.get("name", ""), p.get("arguments") or {})
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    except Exception as e:  # noqa: BLE001
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)[:200]}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def main() -> int:
    out = sys.stdout.buffer
    for raw in sys.stdin.buffer:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            out.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}).encode() + b"\n")
            out.flush()
            continue
        for m in msg if isinstance(msg, list) else [msg]:
            r = handle(m) if isinstance(m, dict) else None
            if r is not None:
                out.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n")
                out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
