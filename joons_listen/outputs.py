"""결과 출력: JSON, TXT, Markdown, SRT, 회의록 Markdown."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        sec, ms = int(sec) + 1, 0
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def speaker_label(result: Dict, sid: str) -> str:
    for s in result.get("speakers", []):
        if s["id"] == sid:
            return s.get("label", sid)
    return sid


def to_txt(result: Dict) -> str:
    lines: List[str] = []
    for seg in result["segments"]:
        mark = " ⚠" if "uncertain" in seg.get("flags", []) else ""
        lines.append(f"[{hms(seg['start'])}] {speaker_label(result, seg['speaker'])}: {seg['text']}{mark}")
        if seg.get("text_ko") and seg["text_ko"] != seg["text"]:
            lines.append(f"    → {seg['text_ko']}")
    return "\n".join(lines) + "\n"


def to_srt(result: Dict) -> str:
    out: List[str] = []
    for n, seg in enumerate(result["segments"], 1):
        text = f"{speaker_label(result, seg['speaker'])}: {seg['text']}"
        if seg.get("text_ko") and seg["text_ko"] != seg["text"]:
            text += f"\n{seg['text_ko']}"
        out.append(f"{n}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n{text}\n")
    return "\n".join(out)


def to_md(result: Dict) -> str:
    src = result.get("source", {})
    title = result.get("title") or Path(src.get("path", "녹음")).stem
    lines = [f"# {title}", ""]
    lines.append(f"- 원본: `{Path(src.get('path', '')).name}` (길이 {hms(src.get('duration_sec', 0))}, SHA256 `{str(src.get('sha256', ''))[:12]}…`)")
    lines.append(f"- 엔진: {result.get('engine')} / 언어: {result.get('language')} / 화자 {len(result.get('speakers', []))}명")
    lines.append(f"- 생성: {result.get('created_at')} (JOONS LISTEN {result.get('version', '')})")
    for w in result.get("warnings", []):
        lines.append(f"- 주의: {w}")
    lines.append("")
    if result.get("uncertain"):
        lines += ["## 확인이 필요한 구간", ""]
        for i in result["uncertain"]:
            seg = result["segments"][i]
            lines.append(f"- [{hms(seg['start'])}] {speaker_label(result, seg['speaker'])}: {seg['text'][:80]}")
        lines.append("")
    facts = result.get("facts", [])
    if facts:
        lines += ["## 가격·수량·날짜·코드 후보 (원문 근거 확인 필요)", "", "| 종류 | 값 | 시각 | 화자 |", "|---|---|---|---|"]
        for f in facts[:200]:
            seg = result["segments"][f["segment"]]
            lines.append(f"| {f['type']} | {f['text']} | {hms(seg['start'])} | {speaker_label(result, seg['speaker'])} |")
        lines.append("")
    lines += ["## 전사", ""]
    for seg in result["segments"]:
        mark = " ⚠" if "uncertain" in seg.get("flags", []) else ""
        lines.append(f"**[{hms(seg['start'])}] {speaker_label(result, seg['speaker'])}**{mark}  ")
        lines.append(seg["text"])
        if seg.get("text_ko") and seg["text_ko"] != seg["text"]:
            lines.append(f"> {seg['text_ko']}")
        lines.append("")
    return "\n".join(lines)


def minutes_md(result: Dict) -> str:
    m = result.get("minutes") or {}
    title = m.get("title") or result.get("title") or "회의록"
    src = result.get("source", {})
    lines = [f"# {title} — 회의록", ""]
    lines.append(f"- 일시/원본: `{Path(src.get('path', '')).name}` (길이 {hms(src.get('duration_sec', 0))})")
    lines.append(f"- 참석(화자 구분): {', '.join(s['label'] for s in result.get('speakers', []))}")
    lines.append(f"- 작성: {'로컬 LLM ' + str(m.get('llm')) if m.get('llm') else '자동 요약 없음(로컬 LLM 미연결, 구조화 결과만)'}")
    lines.append("")
    if m.get("summary"):
        lines += ["## 요약", "", str(m["summary"]), ""]
    for key, head in (("decisions", "결정사항"), ("open_questions", "미결 질문")):
        items = m.get(key) or []
        if items:
            lines += [f"## {head}", ""] + [f"- {x}" for x in items] + [""]
    acts = m.get("actions") or []
    if acts:
        lines += ["## 후속 조치", "", "| 담당 | 내용 | 기한 |", "|---|---|---|"]
        for a in acts:
            lines.append(f"| {a.get('who', '')} | {a.get('what', '')} | {a.get('due', '')} |")
        lines.append("")
    figs = [f for f in result.get("facts", []) if f["type"] in ("price", "quantity", "date", "code")]
    if figs:
        lines += ["## 숫자·날짜·코드 (원문 확인 필요)", ""]
        for f in figs[:80]:
            seg = result["segments"][f["segment"]]
            lines.append(f"- {f['type']}: **{f['text']}** — [{hms(seg['start'])}] {speaker_label(result, seg['speaker'])}")
        lines.append("")
    if result.get("uncertain"):
        lines += ["## 불확실 구간", "", ", ".join(f"[{hms(result['segments'][i]['start'])}]" for i in result["uncertain"]), ""]
    return "\n".join(lines)


def write_all(job_dir: Path, result: Dict) -> Dict[str, str]:
    """결과 파일을 쓰고 {형식: 파일명} 을 돌려준다."""
    job_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "json": ("result.json", json.dumps(result, ensure_ascii=False, indent=2)),
        "txt": ("transcript.txt", to_txt(result)),
        "md": ("transcript.md", to_md(result)),
        "srt": ("subtitles.srt", to_srt(result)),
        "minutes": ("minutes.md", minutes_md(result)),
    }
    out: Dict[str, str] = {}
    for fmt, (name, body) in files.items():
        p = job_dir / name
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(p)
        out[fmt] = name
    return out
