"""Cross-check a finished transcription.

1. Reference vote: other transcripts of the same recording (for example a cloud assistant's transcript pasted in)
   decide ensemble conflicts. A candidate wins only when it appears in a reference and the other candidate does not.
2. LLM arbitration (optional): the local LLM picks A/B for conflicts the references could not decide.
3. Notes check: every line of the meeting notes is checked for its numbers, codes and dictionary terms in the
   transcript, and numbers/codes said in the meeting but missing from the notes are listed with timestamps.

Nothing is confirmed automatically: every item stays "to verify" and keeps its evidence (segment and time).
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import ensemble

_PUNCT = re.compile(r"[\s,，。、；;：:！!？?“”\"'‘’（）()\[\]【】《》<>…·~～\-—]+")
_CN_NUMRUN = re.compile(r"[零〇一幺二两三四五六七八九十百千万亿]{2,}")
_CN_DEC = re.compile(r"([零〇一幺二两三四五六七八九十百千万]+)点([零〇一幺二两三四五六七八九]+)")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,6}-?\d{2,10}[A-Za-z]?(?![A-Za-z0-9])")


def normalize(text: str) -> str:
    """For matching only: Arabic digits for Chinese numbers, lower case, no spaces or punctuation."""
    def dec(m: re.Match) -> str:  # 二十二点四 -> 22.4, 零点九六五 -> 0.965 (before digit runs are converted)
        whole = ensemble._cn_int(m.group(1).replace("幺", "一")) if m.group(1) not in ("零", "〇") else 0
        frac = "".join(ensemble._CN_DIGIT.get(c, "") for c in m.group(2))
        return f"{whole}.{frac}" if whole is not None and frac else m.group(0)

    t = _CN_DEC.sub(dec, text or "")
    t = ensemble.digits_to_arabic(t)

    def unit(m: re.Match) -> str:
        v = ensemble._cn_int(m.group(0).replace("幺", "一"))
        return str(v) if v is not None and v >= 10 else m.group(0)

    t = _CN_NUMRUN.sub(unit, t)
    return _PUNCT.sub("", t).lower()


# ---------- 1) reference vote ----------
def _swap(text: str, old: str, new: str) -> Optional[str]:
    if old and old in text:
        return text.replace(old, new, 1)
    o = old.replace(" ", "")
    if o and o in text:
        return text.replace(o, new, 1)
    return None


def apply_choice(seg: Dict, c: Dict, choice: str, by: str) -> bool:
    """Switch one conflict of a segment to `choice` ('primary'|'secondary'). Returns True when the text changed."""
    if choice == c.get("pick"):
        c["resolved_by"] = by
        return False
    cur, new = (c["secondary"], c["primary"]) if c.get("pick") == "secondary" else (c["primary"], c["secondary"])
    if not cur and not new:
        return False
    out = _swap(seg.get("text", ""), cur, new) if cur else None
    if out is None:
        return False
    seg["text"] = out
    c["pick"] = choice
    c["resolved_by"] = by
    return True


def reference_vote(segments: List[Dict], refs: Sequence[Dict]) -> Dict:
    """refs: [{"name": str, "text": str}]. Decides conflicts where exactly one candidate appears in a reference."""
    nrefs = [(r.get("name") or f"ref{i + 1}", normalize(r.get("text", ""))) for i, r in enumerate(refs) if r.get("text")]
    stats = {"checked": 0, "confirmed": 0, "switched": 0, "undecided": 0}
    if not nrefs:
        return stats
    for seg in segments:
        for c in seg.get("conflicts", []):
            p, s = normalize(c.get("primary", "")), normalize(c.get("secondary", ""))
            if len(p) < 2 and len(s) < 2:
                continue
            stats["checked"] += 1
            vp = [n for n, t in nrefs if len(p) >= 2 and p in t]
            vs = [n for n, t in nrefs if len(s) >= 2 and s in t]
            if vs and not vp:
                choice, names = "secondary", vs
            elif vp and not vs:
                choice, names = "primary", vp
            else:
                stats["undecided"] += 1
                continue
            before = c.get("pick")
            if apply_choice(seg, c, choice, "reference:" + ",".join(names)):
                stats["switched"] += 1
            elif before == choice:
                stats["confirmed"] += 1
            else:  # the span could not be located in the text any more
                c.pop("resolved_by", None)
                stats["undecided"] += 1
    return stats


# ---------- 2) LLM arbitration ----------
def llm_arbitrate(segments: List[Dict], terms: Iterable[str], chat: Callable[[List[Dict]], str], limit: int = 60, batch: int = 15) -> Dict:
    """Ask the local LLM to pick A/B for undecided conflicts. `chat(messages) -> str` returns the model's JSON text."""
    todo = [(seg, c) for seg in segments for c in seg.get("conflicts", [])
            if not c.get("resolved_by") and c.get("primary") and c.get("secondary")][:limit]
    stats = {"asked": len(todo), "switched": 0, "kept": 0, "unsure": 0}
    tl = ", ".join(list(dict.fromkeys(t for t in terms if t))[:80])
    sys = ("You check a meeting transcript. For each item pick the reading that is more likely what was said, "
           "using the sentence context and the glossary. Answer A, B or ? (unsure). Never invent a third reading. "
           'Output JSON only: {"answers":[{"id":int,"pick":"A"|"B"|"?"}]}')
    for b in range(0, len(todo), batch):
        part = todo[b: b + batch]
        items = [{"id": k, "sentence": seg.get("text", "")[:240], "A": c["primary"], "B": c["secondary"]} for k, (seg, c) in enumerate(part)]
        user = f"Glossary: {tl}\nItems:\n" + json.dumps(items, ensure_ascii=False)
        try:
            raw = chat([{"role": "system", "content": sys}, {"role": "user", "content": user}])
            ans = {int(a["id"]): str(a.get("pick", "?")).upper() for a in json.loads(raw).get("answers", [])}
        except Exception:  # noqa: BLE001  a bad answer leaves the items undecided
            ans = {}
        for k, (seg, c) in enumerate(part):
            pick = ans.get(k, "?")
            if pick == "A":
                apply_choice(seg, c, "primary", "llm")
                stats["kept"] += 1
            elif pick == "B":
                stats["switched"] += 1 if apply_choice(seg, c, "secondary", "llm") else 0
            else:
                stats["unsure"] += 1
    return stats


# ---------- 3) notes check ----------
def _keys(text: str, terms: Iterable[str]) -> List[str]:
    """Numbers (2+ digits or decimals), product codes and dictionary terms found in a text, normalized."""
    n = normalize(text)
    keys = [k for k in _NUM.findall(n) if len(k.replace(".", "")) >= 2]
    keys += [m.group(0).lower().replace("-", "") for m in _CODE.finditer(ensemble.digits_to_arabic(text))]
    for t in terms:
        tn = normalize(t)
        if len(tn) >= 2 and not tn.isdigit() and tn in n:
            keys.append(tn)
    return list(dict.fromkeys(keys))


def _hms(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def notes_check(segments: List[Dict], notes: str, terms: Iterable[str]) -> Dict:
    terms = list(terms)
    seg_norm = [(s, normalize(s.get("text", ""))) for s in segments]

    def where(key: str) -> List[str]:
        return [_hms(s["start"]) for s, t in seg_norm if key in t][:3]

    lines = []
    note_keys = set()
    for raw in re.split(r"[\n\r]+|(?<=[。；;])", notes or ""):
        line = raw.strip()
        if len(line) < 4:
            continue
        ks = _keys(line, terms)
        if not ks:
            continue
        note_keys.update(ks)
        found = {k: where(k) for k in ks}
        missing = [k for k, w in found.items() if not w]
        lines.append({"line": line[:200], "found": {k: w for k, w in found.items() if w}, "missing": missing,
                      "status": "ok" if not missing else ("partial" if len(missing) < len(ks) else "not_found")})
    said: Dict[str, Dict] = {}
    for s, t in seg_norm:
        for k in _keys(s.get("text", ""), terms):
            if k in note_keys or k.isalpha():
                continue
            e = said.setdefault(k, {"key": k, "count": 0, "at": [], "context": s.get("text", "")[:120]})
            e["count"] += 1
            if len(e["at"]) < 3:
                e["at"].append(_hms(s["start"]))
    only_said = sorted(said.values(), key=lambda e: -e["count"])[:40]
    return {"lines": lines, "missing_from_notes": only_said,
            "summary": {"note_lines": len(lines), "ok": sum(1 for l in lines if l["status"] == "ok"),
                        "partial": sum(1 for l in lines if l["status"] == "partial"),
                        "not_found": sum(1 for l in lines if l["status"] == "not_found"), "said_not_in_notes": len(only_said)}}


def report_md(result: Dict) -> str:
    cc = result.get("crosscheck") or {}
    out = [f"# 교차 확인: {result.get('title', '')}", ""]
    v, a = cc.get("reference_vote"), cc.get("llm")
    if v:
        out += [f"- 참고 전사 투표: 확인 {v['checked']}건, 바꿈 {v['switched']}건, 판단 못함 {v['undecided']}건"]
    if a:
        out += [f"- 로컬 LLM 판정: 물음 {a['asked']}건, 바꿈 {a['switched']}건, 유지 {a['kept']}건, 모름 {a['unsure']}건"]
    n = cc.get("notes")
    if n:
        s = n["summary"]
        out += ["", "## 회의록 줄별 확인", "", f"줄 {s['note_lines']}개: 일치 {s['ok']}, 일부 {s['partial']}, 녹음에서 못 찾음 {s['not_found']}", ""]
        for l in n["lines"]:
            mark = {"ok": "일치", "partial": "일부", "not_found": "못 찾음"}[l["status"]]
            ev = ", ".join(f"{k}@{'/'.join(w)}" for k, w in l["found"].items())
            miss = (" / 녹음에 없음: " + ", ".join(l["missing"])) if l["missing"] else ""
            out.append(f"- [{mark}] {l['line']}  ({ev}{miss})")
        out += ["", "## 녹음에는 있고 회의록에 없는 숫자·코드", ""]
        for e in n["missing_from_notes"]:
            out.append(f"- {e['key']} ×{e['count']} @{'/'.join(e['at'])}: {e['context']}")
    out.append("")
    return "\n".join(out)
