"""명령줄: serve(서비스), check(환경 점검), run(서비스 없이 파일 하나 직접 처리), sync(미러 상태)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from . import config


def _http(method: str, path: str, body=None, port: int = config.PORT):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"}, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def cmd_serve(a) -> int:
    from .server import serve

    if sys.stdout is None or sys.stderr is None:  # pythonw(창 없는 실행)에는 출력 자리가 없으므로 로그 파일로 돌린다
        config.ensure_dirs()
        f = open(config.DATA_DIR / "service.log", "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = f

    serve(port=a.port)
    return 0


def cmd_check(a) -> int:
    from . import engines, llm

    out = {"version": config.VERSION, "gpu": engines.gpu_info(), "engines": engines.EngineManager().status(), "data_dir": str(config.DATA_DIR), "llm": llm.available()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_run(a) -> int:
    from . import jobs

    try:  # 서비스가 떠 있으면 같은 작업 DB 를 둘이 쓰게 되므로 직접 실행을 막는다
        _http("GET", "/health", port=config.PORT)
        print("The service is already running. Submit the job over HTTP or the MCP server (listen_transcribe); run directly only after stopping the service.", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        pass
    jm = jobs.JobManager()
    jm.start()
    jid = jm.submit(a.kind, {"audio_path": a.audio, "language": a.language, "engine": a.engine, "diarize": not a.no_diarize, "translate_ko": a.translate, "minutes": a.minutes, "num_speakers": a.speakers, "title": a.title, "customer": a.customer})
    last = ""
    while True:
        j = jm.get(jid)
        line = f"{j['state']} {j['stage']} {j['progress']}%"
        if line != last:
            print(line, flush=True)
            last = line
        if j["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(2)
    jm.shutdown()
    print(json.dumps(j, ensure_ascii=False, indent=2))
    print("Result folder:", config.JOBS_DIR / jid)
    return 0 if j["state"] == "done" else 1


def cmd_sync(a) -> int:
    print(json.dumps(_http("GET", "/health", port=a.port).get("store"), ensure_ascii=False))
    return 0


def cmd_mcp(a) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="joons_listen")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=config.PORT)
    s.set_defaults(f=cmd_serve)
    c = sub.add_parser("check")
    c.set_defaults(f=cmd_check)
    r = sub.add_parser("run")
    r.add_argument("audio")
    r.add_argument("--kind", default="transcribe", choices=["transcribe", "diarize"])
    r.add_argument("--language", default="auto")
    r.add_argument("--engine", default="auto", choices=["auto", "whisper", "qwen", "ensemble"])
    r.add_argument("--customer", default=None, help="customer dictionary name (<data>/dictionaries/<name>.json)")
    r.add_argument("--no-diarize", action="store_true")
    r.add_argument("--translate", action="store_true")
    r.add_argument("--minutes", action="store_true")
    r.add_argument("--speakers", type=int, default=None)
    r.add_argument("--title", default=None)
    r.set_defaults(f=cmd_run)
    m = sub.add_parser("mcp", help="MCP server over stdio")
    m.set_defaults(f=cmd_mcp)
    y = sub.add_parser("sync")
    y.add_argument("--port", type=int, default=config.PORT)
    y.set_defaults(f=cmd_sync)
    a = p.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
