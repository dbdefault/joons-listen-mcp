"""전사 파이프라인: 입력 → VAD → 화자 분리 → ASR → 검증 → 번역 → 회의록 → 출력. 단계마다 체크포인트를 남겨 중단 뒤 이어 한다."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import audio, config, dictionary, diarize, engines, ensemble, llm, outputs, vad, verify


class Cancelled(Exception):
    pass


class JobError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Ctx:
    """파이프라인이 job 관리자와 주고받는 이음새."""

    def __init__(self, job_id: str, job_dir: Path, params: Dict, set_stage: Callable[[str, float], None], is_cancelled: Callable[[], bool], warn: Callable[[str], None]):
        self.job_id, self.dir, self.params = job_id, job_dir, params
        self.set_stage, self.is_cancelled, self.warn = set_stage, is_cancelled, warn

    def check(self) -> None:
        if self.is_cancelled():
            raise Cancelled()


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _jsave(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _single_speaker_turns(segs: List[Tuple[float, float]]) -> List[diarize.Turn]:
    return diarize.segments_to_turns(segs, ["S1"] * len(segs))


def run(ctx: Ctx, mgr: engines.EngineManager) -> Dict:
    p = ctx.params
    kind = p.get("kind", "transcribe")
    src = Path(p["audio_path"])
    wav = ctx.dir / "audio.wav"

    # 1) ingest
    ctx.set_stage("ingest", 0.0)
    meta = _jload(ctx.dir / "ingest.json")
    if not meta or not wav.exists():
        dur = audio.to_wav16k(src, wav)
        meta = {"duration_sec": dur, "sha256": audio.sha256_file(src)}
        _jsave(ctx.dir / "ingest.json", meta)
    ctx.check()
    arr = audio.read_wav(wav)

    # 2) vad
    ctx.set_stage("vad", 0.0)
    segs = _jload(ctx.dir / "vad.json")
    if segs is None:
        segs = [list(x) for x in vad.speech_segments(arr)]
        _jsave(ctx.dir / "vad.json", segs)
    ctx.check()
    segs_t = [(float(a), float(b)) for a, b in segs]
    if not segs_t:
        raise JobError("AUDIO_DECODE_FAILED", "no speech found (silence or a very quiet recording)")

    # 3) diarize
    ctx.set_stage("diarize", 0.0)
    turns_j = _jload(ctx.dir / "turns.json")
    if turns_j is None:
        if p.get("diarize", True):
            turns, _info = diarize.diarize(arr, segs_t, p.get("num_speakers"), lambda f: ctx.set_stage("diarize", f))
            diarize.release()
        else:
            turns = _single_speaker_turns(segs_t)
        turns_j = [t.to_dict() for t in turns]
        _jsave(ctx.dir / "turns.json", turns_j)
    ctx.check()

    speakers = _speaker_table(turns_j)
    base = {
        "version": config.VERSION,
        "job_id": ctx.job_id,
        "title": p.get("title") or src.stem,
        "source": {"path": str(src), "sha256": meta["sha256"], "duration_sec": round(meta["duration_sec"], 2)},
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "speakers": speakers,
        "warnings": [],
    }
    if kind == "diarize":
        base.update({"engine": None, "language": None, "segments": [{"i": i, "start": t["start"], "end": t["end"], "speaker": t["speaker"], "lang": None, "text": "", "text_ko": None, "confidence": None, "flags": [], "audio": {"start": t["start"], "end": t["end"]}} for i, t in enumerate(turns_j)], "facts": [], "uncertain": [], "minutes": None})
        return base

    # 4) asr (직전 작업의 LLM 이 GPU 를 쥐고 있으면 먼저 비운다)
    try:
        llm.unload()
    except Exception:  # noqa: BLE001
        pass
    dic = dictionary.load_for(p.get("customer"))
    hints = dictionary.hint_words(dic)
    rules = dictionary.merge_job_replace(dic, p.get("dictionary"))
    asr_path = ctx.dir / "asr.jsonl"
    done = _load_done(asr_path)
    ctx.set_stage("asr", 0.0)
    req_lang = p.get("language", "auto")
    req_lang = None if req_lang in (None, "", "auto") else req_lang
    engine_pref = p.get("engine", "auto") or "auto"
    ens = engine_pref == "ensemble"
    alt: Dict[int, Dict] = {}
    # Merge evidence: with a customer, only that customer's terms count. Global terms (other customers' names) are
    # also given to Whisper as hotwords, and Whisper echoes them on unclear audio (field case: '欧洲保金佳' twice).
    ev_terms = hints
    if ens and p.get("customer"):
        ev_terms = dictionary.hint_words(dictionary.load(dictionary.customer_path(p["customer"]))) or hints
    if ens:
        # Same turns, two engines one after the other (one model on the 8 GB GPU at a time). Qwen is primary.
        _, doc_lang = _run_asr(ctx, mgr, arr, turns_j, done, asr_path, req_lang, "qwen", hints, stage=(0.0, 0.5))
        alt_path = ctx.dir / "asr_whisper.jsonl"
        alt = _load_done(alt_path)
        wl = doc_lang if doc_lang in ("zh", "ko", "en", "ja") else None
        _run_asr(ctx, mgr, arr, turns_j, alt, alt_path, wl, "whisper", hints, stage=(0.5, 1.0))
        chosen = "ensemble(qwen+whisper)"
    else:
        chosen, doc_lang = _run_asr(ctx, mgr, arr, turns_j, done, asr_path, req_lang, engine_pref, hints)
    mgr.unload_all()
    ctx.check()

    segments: List[Dict] = []
    agrees: List[float] = []
    for i, t in enumerate(turns_j):
        r = done[i]
        raw = r.get("text", "")
        echo = dictionary.is_hint_echo(raw, hints)
        text = "" if echo else dictionary.apply_replace(raw, rules).strip()
        flags = ["hint_echo"] if echo else []
        seg = {"i": i, "start": t["start"], "end": t["end"], "speaker": t["speaker"], "lang": r.get("lang"), "text": text, "text_ko": None, "confidence": r.get("conf"), "flags": flags, "audio": {"start": t["start"], "end": t["end"]}}
        if ens:
            w_raw = alt.get(i, {}).get("text", "")
            w_text = "" if dictionary.is_hint_echo(w_raw, hints) else dictionary.apply_replace(w_raw, rules).strip()
            m = ensemble.merge(text, w_text, ev_terms)
            seg["text"] = m["text"] if not echo else ""
            seg["alts"] = {"qwen": text, "whisper": w_text}
            seg["agree"] = m["agree"]
            seg["conflicts"] = m["conflicts"][:20]
            if (text or w_text) and m["agree"] < 0.6:
                flags.append("engines_disagree")
            if text or w_text:
                agrees.append(m["agree"])
        segments.append(seg)
    if ens:
        base["ensemble"] = {"engines": ["qwen", "whisper"], "primary": "qwen", "mean_agree": round(sum(agrees) / len(agrees), 3) if agrees else None,
                            "segments_disagree": sum(1 for s in segments if "engines_disagree" in s["flags"]),
                            "picked_secondary": sum(1 for s in segments for c in s.get("conflicts", []) if c["pick"] == "secondary")}

    # 5) verify
    ctx.set_stage("verify", 0.0)
    ana = verify.analyze(segments)
    for seg in segments:
        for nm in dictionary.name_hits(seg["text"], dic):
            ana["facts"].append({"type": "name", "text": nm, "segment": seg["i"], "verify": True})
    base.update({"engine": chosen, "language": doc_lang, "segments": segments, "facts": ana["facts"], "uncertain": ana["uncertain"], "minutes": None})

    model = None
    if p.get("translate_ko") or p.get("minutes"):
        model = llm.available()
        if not model:
            base["warnings"].append("No local LLM (Ollama) was available, so translation and minutes were skipped (the transcript and structured results are fine). Install a model and run listen_summarize later.")
            ctx.warn("LLM_UNAVAILABLE")

    # 6) translate
    ctx.set_stage("translate", 0.0)
    if p.get("translate_ko") and model:
        idx = [s["i"] for s in segments if s["text"] and s.get("lang") != "ko"]
        tr = llm.translate_lines([segments[i]["text"] for i in idx], model)
        for k, i in enumerate(idx):
            segments[i]["text_ko"] = llm.fix_cny(tr[k], segments[i].get("lang")) if tr[k] else tr[k]
        miss = sum(1 for v in tr if v is None)
        if miss:
            base["warnings"].append(f"{miss} translations were skipped because the model did not answer in the expected format; the original text is kept.")
    ctx.check()

    # 7) minutes
    ctx.set_stage("minutes", 0.0)
    if p.get("minutes") and model:
        base["minutes"] = _minutes(base, model)
    if model:
        llm.unload(model)
    ctx.check()
    return base


def _speaker_table(turns: List[Dict]) -> List[Dict]:
    tot: Dict[str, float] = {}
    for t in turns:
        tot[t["speaker"]] = tot.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    out = []
    for sid in sorted(tot, key=lambda s: int(s[1:])):
        out.append({"id": sid, "label": f"화자 {sid[1:]}", "total_sec": round(tot[sid], 1)})
    return out


def _minutes(result: Dict, model: str) -> Optional[Dict]:
    lines = []
    for s in result["segments"]:
        txt = s.get("text_ko") or s["text"]
        if txt:
            lines.append(f"[{outputs.hms(s['start'])}] {outputs.speaker_label(result, s['speaker'])}: {txt}")
    facts = [f"{f['type']}: {f['text']}" for f in result.get("facts", []) if f["type"] != "name"]
    try:
        return llm.make_minutes(lines, facts, result.get("title", "회의"), model)
    except Exception as e:  # noqa: BLE001
        result["warnings"].append(f"minutes generation failed: {str(e)[:120]}")
        return None


def summarize_existing(ctx: Ctx, parent_result: Dict) -> Dict:
    """이미 끝난 전사 결과로 번역·회의록만 다시 만든다(kind=summarize)."""
    res = json.loads(json.dumps(parent_result))
    res["job_id"] = ctx.job_id
    res.setdefault("warnings", [])
    ctx.set_stage("minutes", 0.0)
    model = llm.available()
    if not model:
        raise JobError("LLM_UNAVAILABLE", "the local LLM (Ollama) has no usable model (for example: ollama pull qwen3:8b)")
    if ctx.params.get("translate_ko"):
        idx = [s["i"] for s in res["segments"] if s["text"] and s.get("lang") != "ko"]
        tr = llm.translate_lines([res["segments"][i]["text"] for i in idx], model)
        for k, i in enumerate(idx):
            res["segments"][i]["text_ko"] = llm.fix_cny(tr[k], res["segments"][i].get("lang")) if tr[k] else tr[k]
    res["minutes"] = _minutes(res, model)
    llm.unload(model)
    return res


# ---------- ASR 실행 ----------
def _detect_language(mgr: engines.EngineManager, arr: np.ndarray, turns: List[Dict], hints) -> Tuple[Optional[str], Dict[int, Dict]]:
    """언어를 모를 때: 가장 긴 발언 몇 개를 Qwen3-ASR(자동 언어 인식)에 먼저 넣어 대표 언어를 정한다. 그 결과는 재사용한다."""
    cand = sorted(range(len(turns)), key=lambda i: -(turns[i]["end"] - turns[i]["start"]))
    cand = [i for i in cand if turns[i]["end"] - turns[i]["start"] >= 1.5][:4] or cand[:3]
    q = mgr.use("qwen")
    res = q.transcribe_batch([audio.slice_audio(arr, turns[i]["start"], turns[i]["end"]) for i in cand], None, hints)
    reuse = {i: {"i": i, "text": r["text"], "lang": r["language"], "conf": r["confidence"]} for i, r in zip(cand, res)}
    return engines.majority_language([r["language"] for r in res]), reuse


def _load_done(path: Path) -> Dict[int, Dict]:
    done: Dict[int, Dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:  # 강제 종료로 마지막 줄이 잘렸으면 그 줄만 버리고 해당 발언은 다시 전사한다
                r = json.loads(line)
                done[int(r["i"])] = r
            except (ValueError, KeyError, TypeError):
                continue
    return done


def _run_asr(ctx: Ctx, mgr: engines.EngineManager, arr: np.ndarray, turns: List[Dict], done: Dict[int, Dict], asr_path: Path, req_lang: Optional[str], engine_pref: str, hints, stage: Tuple[float, float] = (0.0, 1.0)) -> Tuple[str, Optional[str]]:
    lo, hi = stage

    def prog(frac: float) -> None:
        ctx.set_stage("asr", lo + (hi - lo) * frac)

    reuse: Dict[int, Dict] = {}
    lang = req_lang
    if lang is None and engine_pref == "auto":
        lang, reuse = _detect_language(mgr, arr, turns, hints)
    chosen = engines.pick_engine(lang if lang != "mixed" else "mixed", engine_pref)
    total = len(turns)

    def commit(rec: Dict) -> None:
        done[rec["i"]] = rec
        with open(asr_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if chosen == "qwen":
        for i, r in reuse.items():
            if i not in done:
                commit(r)
    todo = [i for i in range(total) if i not in done]
    eng = mgr.use(chosen)
    force = lang if lang in ("zh", "ko", "en", "ja") else None
    n_done = total - len(todo)
    prog(n_done / total)
    if chosen == "whisper":
        for i in todo:
            ctx.check()
            r = eng.transcribe(audio.slice_audio(arr, turns[i]["start"], turns[i]["end"]), force, hints)
            commit({"i": i, "text": r["text"], "lang": r["language"], "conf": r["confidence"]})
            n_done += 1
            prog(n_done / total)
    else:
        B = max(1, config.QWEN_BATCH)
        for b in range(0, len(todo), B):
            ctx.check()
            grp = todo[b : b + B]
            res = eng.transcribe_batch([audio.slice_audio(arr, turns[i]["start"], turns[i]["end"]) for i in grp], force, hints)
            for i, r in zip(grp, res):
                commit({"i": i, "text": r["text"], "lang": r["language"], "conf": r["confidence"]})
            n_done += len(grp)
            prog(n_done / total)
    doc_lang = lang
    if doc_lang is None:
        doc_lang = engines.majority_language([d.get("lang") for d in done.values()])
    return chosen, doc_lang
