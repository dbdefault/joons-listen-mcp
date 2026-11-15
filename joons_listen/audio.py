"""오디오 입력: ffmpeg 로 16kHz 모노 WAV 로 바꾸고, 길이·해시·구간 자르기를 돕는다."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 16000


class AudioError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def find_ffmpeg() -> str:
    exe = os.environ.get("JOONS_LISTEN_FFMPEG") or shutil.which("ffmpeg")
    if exe and Path(exe).exists():
        return exe
    guess = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
    if guess.exists():
        for p in guess.glob("ffmpeg*/bin/ffmpeg.exe"):
            return str(p)
    raise AudioError("AUDIO_DECODE_FAILED", "ffmpeg 를 찾지 못했습니다(JOONS_LISTEN_FFMPEG 또는 PATH 확인)")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def to_wav16k(src: Path, dst: Path, timeout: int = 3600) -> float:
    """원본을 16kHz 모노 PCM WAV 로 변환하고 길이(초)를 돌려준다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise AudioError("AUDIO_DECODE_FAILED", "오디오 변환이 시간 초과되었습니다")
    if r.returncode != 0 or not dst.exists():
        msg = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-1:] or ["알 수 없는 오류"]
        raise AudioError("AUDIO_DECODE_FAILED", f"오디오를 읽을 수 없습니다: {msg[0][:200]}")
    with wave.open(str(dst), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
    if dur <= 0.05:
        raise AudioError("AUDIO_DECODE_FAILED", "오디오 길이가 0 입니다")
    return dur


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SR:
        raise AudioError("AUDIO_DECODE_FAILED", f"샘플레이트가 {SR} 가 아닙니다({sr})")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def slice_audio(arr: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * SR)))
    b = min(len(arr), int(round(end * SR)))
    return arr[a:b]


def cut_clip(src_wav: Path, start: float, end: float, dst: Path) -> None:
    """원본 음성과 전사 문장 연결용 구간 잘라내기(WAV)."""
    import soundfile as sf

    arr = read_wav(src_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), slice_audio(arr, start, end), SR, subtype="PCM_16")
