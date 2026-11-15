"""음성 구간 검출(silero-vad)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import config

_model = None


def _load():
    global _model
    if _model is None:
        from silero_vad import load_silero_vad

        _model = load_silero_vad()
    return _model


def speech_segments(audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float]]:
    """[(시작초, 끝초)]. 한 구간은 TURN_MAX_SEC 안으로 침묵에서 잘린다."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _load()
    wav = torch.from_numpy(audio)
    ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        return_seconds=True,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        max_speech_duration_s=config.TURN_MAX_SEC - 3,
    )
    out = [(float(t["start"]), float(t["end"])) for t in ts]
    out = [(s, e) for s, e in out if e - s >= 0.2]
    return fill_gaps(out, audio, sr)


def fill_gaps(segs: List[Tuple[float, float]], audio: np.ndarray, sr: int = 16000, min_gap: float = 30.0, chunk: float = 25.0, min_db: float = -50.0) -> List[Tuple[float, float]]:
    """silero 가 놓친 긴 빈 구간(원거리 녹음·잡음 속 대화)을 고정 길이 조각으로 메운다.
    소리가 거의 없는(min_db 미만) 조각은 환각을 막으려고 넣지 않는다. 실측: 56분 회의에서 silero 가 21분을 통째로 놓침."""
    total = len(audio) / sr
    merged: List[Tuple[float, float]] = []
    prev = 0.0
    for s, e in sorted(segs) + [(total, total)]:
        gap = s - prev
        if gap >= min_gap:
            t = prev
            while t < s - 1.0:
                u = min(t + chunk, s)
                seg = audio[int(t * sr) : int(u * sr)]
                if len(seg) and 20 * np.log10(float(np.sqrt(np.mean(np.square(seg.astype(np.float64))))) + 1e-9) > min_db:
                    merged.append((t, u))
                t = u
        if e > s:
            merged.append((s, e))
        prev = max(prev, e)
    return sorted(merged)
