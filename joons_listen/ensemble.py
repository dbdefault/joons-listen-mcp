"""Cross-engine merge: combine two ASR hypotheses of the same segment.

The primary engine (Qwen3-ASR for zh/ko) keeps the final say unless the secondary engine (Whisper) has a
dictionary term or a number in the span where the two disagree. Nothing is invented: every output token comes
from one of the two hypotheses. Spans present in only one engine follow the primary (secondary-only text is a
frequent hallucination source), and every disagreement is reported so a later step (or a person) can review it.
"""
from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Sequence

_CN_DIGIT = {"零": "0", "〇": "0", "一": "1", "幺": "1", "二": "2", "两": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
# Digit-by-digit readings such as 六五零六 (6506) or 一四五零 (1450). Readings with units (十, 百, 千, 万) are left
# alone because "二十二点四" style numbers need real parsing and a wrong guess is worse than no change.
_CN_RUN = re.compile(r"[零〇一幺二两三四五六七八九]{3,}")
_TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\sA-Za-z\d가-힣]")


def digits_to_arabic(text: str) -> str:
    """Convert digit-by-digit Chinese number readings of 3+ characters (model codes, prices) to Arabic digits."""
    return _CN_RUN.sub(lambda m: "".join(_CN_DIGIT[c] for c in m.group(0)), text or "")


def _spans(text: str) -> List[re.Match]:
    return [m for m in _TOKEN.finditer(text or "") if not (len(m.group(0)) == 1 and not m.group(0).isalnum())]


def tokens(text: str) -> List[str]:
    """Chinese/Japanese characters one by one; Korean words, Latin words and numbers as whole tokens; punctuation dropped."""
    return [m.group(0) for m in _spans(text)]


def _key(tok: str) -> str:
    return tok.lower()


def _wordlike(c: str) -> bool:
    return (c.isascii() and c.isalnum()) or ("가" <= c <= "힣")


def _join(toks: Sequence[str]) -> str:
    out = ""
    for t in toks:
        if out and _wordlike(out[-1]) and _wordlike(t[0]):
            out += " "
        out += t
    return out


def _score(span: str, terms: Iterable[str]) -> int:
    s = 0
    low = span.lower()
    for w in terms:
        if w and w.lower() in low:
            s += 3
    if re.search(r"\d", span):
        s += 1
    return s


def merge(primary: str, secondary: str, terms: Iterable[str] = ()) -> Dict:
    """Merge two hypotheses. Returns {"text", "agree", "conflicts": [{"primary","secondary","pick"}]}."""
    terms = [t for t in {str(w).strip() for w in terms} if t]
    pt = digits_to_arabic(primary).strip()
    am = _spans(pt)
    a = [m.group(0) for m in am]
    b = tokens(digits_to_arabic(secondary))
    if not a and not b:
        return {"text": "", "agree": 1.0, "conflicts": []}
    if not b:
        return {"text": pt, "agree": 0.0, "conflicts": [{"primary": _join(a), "secondary": "", "pick": "primary"}]}
    if not a:
        return {"text": pt, "agree": 0.0, "conflicts": [{"primary": "", "secondary": _join(b), "pick": "primary"}]}
    sm = difflib.SequenceMatcher(None, [_key(t) for t in a], [_key(t) for t in b], autojunk=False)
    same = 0
    conflicts: List[Dict] = []
    swaps: List[tuple] = []  # (start, end, replacement) in the primary string; punctuation of the primary is kept
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            same += i2 - i1
            continue
        pa, sb = a[i1:i2], b[j1:j2]
        pick = "primary"
        if op == "replace":
            # Look a few tokens around the span so a term split across the boundary still counts.
            if _score(_join(b[max(0, j1 - 4): j2 + 4]), terms) > _score(_join(a[max(0, i1 - 4): i2 + 4]), terms):
                pick = "secondary"
                swaps.append((am[i1].start(), am[i2 - 1].end(), _join(sb)))
        conflicts.append({"primary": _join(pa), "secondary": _join(sb), "pick": pick})
    text = pt
    for s, e, rep in reversed(swaps):
        text = text[:s] + rep + text[e:]
    agree = same / max(len(a), len(b))
    return {"text": text, "agree": round(agree, 3), "conflicts": conflicts}
