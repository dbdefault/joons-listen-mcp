"""ASR 엔진 두 개: OpenAI Whisper large-v3(faster-whisper/CTranslate2)와 Alibaba Qwen3-ASR-1.7B(공식 qwen-asr).
GPU(8GB)에는 한 번에 하나만 올린다. 다른 ASR 모델은 쓰지 않는다."""
from __future__ import annotations

import gc
import inspect
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config

LANG_CODE = {
    "chinese": "zh", "mandarin": "zh", "cantonese": "zh", "zh": "zh", "yue": "zh",
    "korean": "ko", "ko": "ko", "english": "en", "en": "en", "japanese": "ja", "ja": "ja",
}
QWEN_LANG_NAME = {"zh": "Chinese", "ko": "Korean", "en": "English", "ja": "Japanese"}


class EngineError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def norm_lang(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    return LANG_CODE.get(str(x).strip().lower())


def _add_cuda_dll_dirs() -> None:
    """torch 가 가져온 CUDA 12 DLL(cuBLAS·cuDNN)을 ctranslate2 가 찾도록 경로에 추가한다."""
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401

        lib = Path(torch.__file__).parent / "lib"
        if lib.exists():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def gpu_info() -> Optional[Dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"name": torch.cuda.get_device_name(0), "vram_total_mb": int(total / 2**20), "vram_free_mb": int(free / 2**20)}
    except Exception:
        return None


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class WhisperEngine:
    id = "whisper"
    name = "OpenAI Whisper large-v3 (faster-whisper / CTranslate2)"
    license = "MIT (OpenAI Whisper), CTranslate2 변환: SYSTRAN"

    def __init__(self) -> None:
        self.model = None
        self.device = None

    def available(self) -> bool:
        return (config.MODELS_DIR / "faster-whisper-large-v3" / "model.bin").exists()

    def load(self) -> None:
        if self.model is not None:
            return
        _add_cuda_dll_dirs()
        try:
            import ctranslate2
            from faster_whisper import WhisperModel
        except Exception as e:  # noqa: BLE001
            raise EngineError("ENGINE_UNAVAILABLE", f"cannot import faster-whisper: {e}")
        local = config.MODELS_DIR / "faster-whisper-large-v3"
        src = str(local) if (local / "model.bin").exists() else config.WHISPER_MODEL
        cuda = ctranslate2.get_cuda_device_count() > 0
        self.device = "cuda" if cuda else "cpu"
        compute = config.WHISPER_COMPUTE if cuda else "int8"
        try:
            self.model = WhisperModel(src, device=self.device, compute_type=compute, download_root=str(config.MODELS_DIR / "hf"))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            code = "GPU_OOM" if "out of memory" in msg.lower() else "ENGINE_UNAVAILABLE"
            raise EngineError(code, f"failed to load Whisper large-v3: {msg[:200]}")

    def unload(self) -> None:
        self.model = None
        _free_gpu()

    def transcribe(self, audio: np.ndarray, language: Optional[str], hints: Sequence[str] = ()) -> Dict:
        self.load()
        assert self.model is not None
        kw = dict(
            language=language,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        if hints:
            kw["hotwords"] = " ".join(hints)[:400]
        try:
            segs, info = self.model.transcribe(audio, **kw)
            parts: List[str] = []
            lp, dur = 0.0, 0.0
            for s in segs:
                parts.append(s.text)
                d = max(0.01, s.end - s.start)
                lp += s.avg_logprob * d
                dur += d
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            raise EngineError("GPU_OOM" if "out of memory" in msg.lower() else "INTERNAL", f"Whisper transcription failed: {msg[:200]}")
        conf = math.exp(lp / dur) if dur > 0 else None
        return {"text": "".join(parts).strip(), "language": norm_lang(info.language) or info.language, "lang_prob": float(info.language_probability or 0), "confidence": conf}


class QwenEngine:
    id = "qwen"
    name = "Alibaba Qwen3-ASR-1.7B (qwen-asr)"
    license = "Apache-2.0"

    def __init__(self) -> None:
        self.model = None
        self._accepts: set = set()

    def available(self) -> bool:
        return (config.MODELS_DIR / "qwen3-asr-1.7b" / "config.json").exists()

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except Exception as e:  # noqa: BLE001
            raise EngineError("ENGINE_UNAVAILABLE", f"cannot import qwen-asr: {e}")
        local = config.MODELS_DIR / "qwen3-asr-1.7b"
        src = str(local) if (local / "config.json").exists() else config.QWEN_MODEL
        kw = dict(dtype=torch.bfloat16, device_map="cuda:0" if torch.cuda.is_available() else "cpu", max_inference_batch_size=config.QWEN_BATCH, max_new_tokens=512)
        if not torch.cuda.is_available():
            kw["dtype"] = torch.float32
        if config.QWEN_ALIGNER:
            kw["forced_aligner"] = config.QWEN_ALIGNER
        try:
            self.model = Qwen3ASRModel.from_pretrained(src, **kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            raise EngineError("GPU_OOM" if "out of memory" in msg.lower() else "ENGINE_UNAVAILABLE", f"failed to load Qwen3-ASR: {msg[:200]}")
        try:
            self._accepts = set(inspect.signature(self.model.transcribe).parameters)
        except Exception:
            self._accepts = {"audio", "language"}

    def unload(self) -> None:
        self.model = None
        _free_gpu()

    def transcribe_batch(self, audios: Sequence[np.ndarray], language: Optional[str], hints: Sequence[str] = ()) -> List[Dict]:
        self.load()
        assert self.model is not None
        lang_arg = QWEN_LANG_NAME.get(language or "", None)
        kw: Dict = {"audio": [(a, 16000) for a in audios], "language": lang_arg}
        if hints and "context" in self._accepts:
            kw["context"] = ", ".join(hints)[:400]
        try:
            res = self.model.transcribe(**kw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            raise EngineError("GPU_OOM" if "out of memory" in msg.lower() else "INTERNAL", f"Qwen3-ASR transcription failed: {msg[:200]}")
        out: List[Dict] = []
        for r in res:
            text = getattr(r, "text", "") or ""
            lang = norm_lang(getattr(r, "language", None))
            out.append({"text": text.strip(), "language": lang or getattr(r, "language", None), "confidence": None})
        return out


class EngineManager:
    def __init__(self) -> None:
        self.whisper = WhisperEngine()
        self.qwen = QwenEngine()
        self.active: Optional[str] = None

    def use(self, which: str):
        if self.active and self.active != which:
            (self.whisper if self.active == "whisper" else self.qwen).unload()
            self.active = None
        eng = self.whisper if which == "whisper" else self.qwen
        eng.load()
        self.active = which
        return eng

    def unload_all(self) -> None:
        self.whisper.unload()
        self.qwen.unload()
        self.active = None

    def status(self) -> Dict:
        return {
            "whisper": {"ready": self.whisper.available(), "loaded": self.whisper.model is not None, "model": "large-v3"},
            "qwen": {"ready": self.qwen.available(), "loaded": self.qwen.model is not None, "model": "Qwen3-ASR-1.7B"},
        }


def pick_engine(language: Optional[str], engine: str = "auto") -> str:
    """초기 기본 라우팅(성능 우열이 확정된 것은 아님): 영어 → Whisper, 중국어·한국어·혼용 → Qwen. 수동 지정이 우선."""
    if engine in ("whisper", "qwen"):
        return engine
    return "whisper" if language == "en" else "qwen"


def majority_language(langs: Sequence[Optional[str]]) -> Optional[str]:
    """여러 발언의 언어 → 대표 언어. 둘 이상이 의미 있게 섞이면 'mixed'."""
    cnt: Dict[str, int] = {}
    for l in langs:
        if l:
            cnt[l] = cnt.get(l, 0) + 1
    if not cnt:
        return None
    total = sum(cnt.values())
    top, n = max(cnt.items(), key=lambda kv: kv[1])
    if n / total >= 0.8:
        return top
    return "mixed"
