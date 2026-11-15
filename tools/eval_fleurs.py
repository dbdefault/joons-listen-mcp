"""시스템 성능 평가(모델끼리 우열 경쟁이 아니라 우리 서비스의 처리 성능 측정).
FLEURS dev 에서 언어별 N개를 고정 시드로 뽑아 두 엔진으로 전사하고 오류율(zh·ko: CER, en: WER), 처리 속도(RTF), GPU 메모리를 잰다.
사용: python tools/eval_fleurs.py --root <데이터 폴더>/eval/fleurs --n 25 --out result.json
(데이터는 google/fleurs 의 data/<lang>/audio/dev.tar.gz 와 dev.tsv, 각 언어 폴더에 풀어 둔다)"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANGS = {"zh": "zh", "ko": "ko", "en": "en"}


_CN_DIG = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_CN_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")


def _cn_to_int(s: str) -> str:
    """중국어 숫자 한 덩어리 → 아라비아 숫자(평가 공정성용: 모델이 '一万九千五百'로 쓰고 정답이 '19500'인 경우)."""
    if not any(c in s for c in "十百千万亿"):
        return "".join(str(_CN_DIG[c]) for c in s) if len(s) > 1 else s
    return str(_cn_value(s))


def _cn_value(s: str) -> int:
    """亿·万 단위를 재귀로 나눠 계산한다('一万亿'은 1조, '三亿五千万'은 3억 5천만)."""
    if "亿" in s:
        left, right = s.rsplit("亿", 1)
        return (_cn_value(left) if left else 1) * 100000000 + (_cn_value(right) if right else 0)
    if "万" in s:
        left, right = s.rsplit("万", 1)
        return (_cn_value(left) if left else 1) * 10000 + (_cn_value(right) if right else 0)
    total, num = 0, 0
    for c in s:
        if c in _CN_DIG:
            num = _CN_DIG[c]
        elif c in _CN_UNIT:
            total += (num or 1) * _CN_UNIT[c]
            num = 0
    return total + num


def norm(text: str, lang: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    if lang == "zh":
        t = _CN_RE.sub(lambda m: _cn_to_int(m.group(0)), re.sub(r"(?<=[一-鿿]) (?=[一-鿿0-9])", "", t))
    t = re.sub(r"[^\w\s]", " ", t)
    t = t.replace("_", " ")
    if lang in ("zh", "ko"):
        t = re.sub(r"\s+", "", t) if lang == "zh" else re.sub(r"\s+", " ", t).strip()
        if lang == "ko":
            t = t.replace(" ", "")
        return t
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(ref: str, hyp: str, lang: str):
    r, h = norm(ref, lang), norm(hyp, lang)
    if lang == "en":
        rt, ht = r.split(), h.split()
    else:
        rt, ht = list(r), list(h)
    return edit_distance(rt, ht), max(1, len(rt))


def load_clips(root: Path, lang: str, n: int, seed: int):
    d = root / lang
    tsv = d / "dev.tsv"
    wavs = {p.name: p for p in d.rglob("*.wav")}
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 4 and r[1] in wavs:
                rows.append((wavs[r[1]], r[3] if r[3].strip() else r[2], r[2]))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--langs", default="zh,ko,en")
    ap.add_argument("--engines", default="whisper,qwen")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch

    from joons_listen import audio, config, engines

    config.ensure_dirs()
    mgr = engines.EngineManager()
    root = Path(a.root)
    tmp = Path(config.DATA_DIR) / "eval" / "_wav16"
    tmp.mkdir(parents=True, exist_ok=True)
    clips = {l: load_clips(root, l, a.n, a.seed) for l in a.langs.split(",")}
    result = {"n_per_lang": a.n, "seed": a.seed, "gpu": engines.gpu_info(), "engines": {}}

    for eng in a.engines.split(","):
        t_load = time.time()
        e = mgr.use(eng)
        load_sec = time.time() - t_load
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # 워밍업: 첫 클립에 섞이는 커널 컴파일·캐시 효과를 속도 측정에서 뺀다(측정에는 쓰지 않는다)
        first = next((it for items in clips.values() for it in items), None)
        if first is not None:
            w0 = tmp / "_warmup.wav"
            audio.to_wav16k(first[0], w0)
            a0 = audio.read_wav(w0)
            lang0 = next(iter(clips))
            if eng == "whisper":
                e.transcribe(a0, LANGS[lang0])
            else:
                e.transcribe_batch([a0], LANGS[lang0])
        er = {"load_sec": round(load_sec, 1), "langs": {}}
        for lang, items in clips.items():
            errs = toks = 0
            dur = proc = 0.0
            samples = []
            for wav, ref, raw in items:
                w16 = tmp / (wav.stem + ".wav")
                audio.to_wav16k(wav, w16)
                arr = audio.read_wav(w16)
                d = len(arr) / 16000
                t0 = time.time()
                if eng == "whisper":
                    hyp = e.transcribe(arr, LANGS[lang])["text"]
                else:
                    hyp = e.transcribe_batch([arr], LANGS[lang])[0]["text"]
                p = time.time() - t0
                x, y = error_rate(ref, hyp, lang)
                errs += x
                toks += y
                dur += d
                proc += p
                if len(samples) < 3:
                    samples.append({"ref": ref[:120], "hyp": hyp[:120]})
            er["langs"][lang] = {"clips": len(items), "metric": "WER" if lang == "en" else "CER", "error_rate": round(errs / toks, 4), "audio_sec": round(dur, 1), "proc_sec": round(proc, 1), "rtf": round(proc / dur, 3), "samples": samples}
        g = engines.gpu_info()
        er["vram_used_mb_after_run"] = (g["vram_total_mb"] - g["vram_free_mb"]) if g else None  # 이 프로세스가 쥔 GPU 메모리(다른 프로그램 사용분 포함)
        if torch.cuda.is_available():
            er["torch_peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)  # torch 가 직접 잡은 최대치(Whisper/CTranslate2 는 이 값에 안 잡힘)
        result["engines"][eng] = er
        mgr.unload_all()
        Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(eng, json.dumps({l: (v["metric"], v["error_rate"], v["rtf"]) for l, v in er["langs"].items()}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
