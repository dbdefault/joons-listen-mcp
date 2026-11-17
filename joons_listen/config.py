"""JOONS LISTEN MCP 설정. 값은 모두 환경변수로 바꿀 수 있다."""
from __future__ import annotations

import os
from pathlib import Path

VERSION = "0.1.0"
SERVICE_NAME = "JOONS LISTEN MCP"

HOME = Path(os.environ.get("USERPROFILE") or Path.home())  # Windows 는 USERPROFILE, 그 밖에는 홈 폴더
DATA_DIR = Path(os.environ.get("JOONS_LISTEN_DATA", str(HOME / ".joons-listen")))
PORT = int(os.environ.get("JOONS_LISTEN_PORT", "18796"))
HOST = os.environ.get("JOONS_LISTEN_HOST", "127.0.0.1")  # 루프백에만 묶는다. 바꾸지 않는다.

INBOX_DIR = DATA_DIR / "inbox"
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "jobs.sqlite3"
DICT_PATH = Path(os.environ.get("JOONS_LISTEN_DICT", str(DATA_DIR / "dictionary.json")))
# Per-customer dictionaries (<name>.json). Keep them out of any public repository.
CUSTOMER_DICT_DIR = Path(os.environ.get("JOONS_LISTEN_DICT_DIR", str(DATA_DIR / "dictionaries")))

# 허용 폴더: 서비스는 이 폴더 안의 오디오 파일만 읽는다(임의 파일 읽기 방지).
_default_roots = [INBOX_DIR, HOME / "Desktop", HOME / "Downloads"]
ALLOWED_ROOTS = [Path(p) for p in os.environ.get("JOONS_LISTEN_ROOTS", "").split(os.pathsep) if p.strip()] or _default_roots
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4", ".mkv", ".webm", ".amr", ".3gp"}
SECRET_NAME_PARTS = ("secret", "token", "credential", "password", "passwd", "api_key", "apikey", "private_key", ".env", ".pem", ".pfx")

# 모델(공식 파일만). 두 개만 쓴다.
WHISPER_MODEL = os.environ.get("JOONS_LISTEN_WHISPER", "large-v3")  # SYSTRAN/faster-whisper-large-v3 (OpenAI Whisper large-v3 의 CTranslate2 변환)
WHISPER_REPO = "Systran/faster-whisper-large-v3"
QWEN_MODEL = os.environ.get("JOONS_LISTEN_QWEN", "Qwen/Qwen3-ASR-1.7B")
QWEN_ALIGNER = os.environ.get("JOONS_LISTEN_QWEN_ALIGNER", "")  # 비어 있으면 강제정렬기를 쓰지 않는다(구간 시각은 화자 구간 기준)
WHISPER_COMPUTE = os.environ.get("JOONS_LISTEN_WHISPER_COMPUTE", "int8_float16")
QWEN_BATCH = int(os.environ.get("JOONS_LISTEN_QWEN_BATCH", "4"))

# 화자 분리(비게이트): silero-vad + speechbrain ECAPA + 군집
ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
DIAR_THRESHOLD = float(os.environ.get("JOONS_LISTEN_DIAR_THRESHOLD", "0.62"))  # 코사인 거리 임계값(작을수록 화자를 더 잘게 나눔)
DIAR_WINDOW_SEC = 3.0
TURN_MAX_SEC = 28.0   # ASR 에 넣는 한 번의 최대 길이
TURN_GAP_SEC = 1.0    # 같은 화자 구간을 이어 붙이는 최대 침묵

# 로컬 LLM(회의록·번역). 외부 클라우드 API 로는 보내지 않는다.
LLM_URL = os.environ.get("JOONS_LISTEN_LLM_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("JOONS_LISTEN_LLM_MODEL", "qwen3:8b")
LLM_TIMEOUT = int(os.environ.get("JOONS_LISTEN_LLM_TIMEOUT", "600"))

# 결과 미러링(선택): JOONS_LISTEN_MIRROR_DIR 에 폴더를 주면 끝난 작업의 결과(원본 오디오 제외)를 zip 으로 그 폴더에 복사한다. 비어 있으면 끈다.
MIRROR_DIR = os.environ.get("JOONS_LISTEN_MIRROR_DIR", "").strip()
# 또는 JOONS_LISTEN_MIRROR_CMD 에 JSON 배열로 명령을 주면 파일마다 그 명령을 실행한다(예: ["rsync", "-a", "{file}", "host:/dir/"]). {file} 은 보낼 파일 경로로 바뀐다.
# 셸을 거치지 않고, zip 을 먼저 보낸 뒤 표식(.ok)을 마지막에 보낸다. 명령이 0 이 아닌 값으로 끝나면 다음 주기에 다시 보낸다.
MIRROR_CMD = os.environ.get("JOONS_LISTEN_MIRROR_CMD", "").strip()
MIRROR_MODE = "cmd" if MIRROR_CMD else ("folder" if MIRROR_DIR else "off")

STAGES = ["ingest", "vad", "diarize", "asr", "verify", "translate", "minutes", "store"]
STAGE_WEIGHTS = {"ingest": 5, "vad": 5, "diarize": 15, "asr": 55, "verify": 3, "translate": 7, "minutes": 7, "store": 3}


def ensure_dirs() -> None:
    for d in (DATA_DIR, INBOX_DIR, JOBS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # 모델 캐시를 DATA_DIR 안에 모아 둔다(지울 때 한 폴더).
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
