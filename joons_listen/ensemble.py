"""Cross-engine merge: combine two ASR hypotheses of the same segment.

The primary engine (Qwen3-ASR for zh/ko) keeps the final say unless the secondary engine (Whisper) has a
dictionary term inside the span where the two disagree (and the primary span has none), and the two spans are
of similar length. Nothing is invented: every output token comes from one of the two hypotheses. Spans present
in only one engine follow the primary (secondary-only text is a frequent hallucination source), and every
disagreement is reported so a later step (or a person) can review it.

Numbers get no bonus: field recordings showed Whisper's Arabic digits are often a different number or a
fragment. A disagreement where both sides are the same number (二十 vs 20) counts as agreement.
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


_CN_NUM = {"零": 0, "〇": 0, "一": 1, "幺": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


def _cn_int(s: str):
    """十二 -> 12, 一千六百 -> 1600, 一万一千 -> 11000. None when s is not a plain Chinese number."""
    if not s or any(c not in _CN_NUM and c not in _CN_UNIT for c in s):
        return None
    total, section, num = 0, 0, None
    for c in s:
        if c in _CN_NUM:
            num = _CN_NUM[c]
        else:
            u = _CN_UNIT[c]
            if u >= 10000:
                total += (section + (num or 0)) * u
                section = 0
            else:
                section += (1 if num is None else num) * u
            num = None
    return total + section + (num or 0)


def _same_number(pa: str, sb: str) -> bool:
    """The two disagreeing spans are the same number written differently (二十 vs 20, 一千六百 vs 1600)."""
    digits = re.sub(r"[^\d]", "", sb)
    if not digits or len(digits) > 12:
        return False
    v = _cn_int(re.sub(r"[^零〇一幺二两三四五六七八九十百千万亿]", "", pa))
    return v is not None and str(v) == digits


def _term_hits(text: str, lo: int, hi: int, terms: Iterable[str]) -> int:
    """Dictionary terms in `text` that overlap the character range [lo, hi)."""
    n = 0
    low = text.lower()
    for w in terms:
        wl = w.lower()
        start = low.find(wl)
        while start >= 0:
            if start < hi and start + len(wl) > lo:
                n += 1
                break
            start = low.find(wl, start + 1)
    return n


def _ctx(toks: Sequence[str], i1: int, i2: int):
    """Joined text of a few tokens around [i1, i2) and the character range of the span inside it."""
    left = _join(toks[max(0, i1 - 4): i1])
    mid = _join(toks[i1:i2])
    full = _join(list(toks[max(0, i1 - 4): i2 + 4]))
    lo = full.find(mid, max(0, len(left) - 1)) if mid else len(left)
    lo = len(left) if lo < 0 else lo
    return full, lo, lo + len(mid)


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
        ta, tb = _join(pa), _join(sb)
        if op == "replace" and _same_number(ta, tb):
            same += i2 - i1
            continue
        pick = "primary"
        if op == "replace":
            # A term that only partly overlaps the span (split across the boundary) still counts, via the context.
            ca, la, ha = _ctx(a, i1, i2)
            cb, lb, hb = _ctx(b, j1, j2)
            la_len, lb_len = len(ta.replace(" ", "")), len(tb.replace(" ", ""))
            similar = 0.5 <= lb_len / max(1, la_len) <= 2.0
            if similar and _term_hits(cb, lb, hb, terms) > _term_hits(ca, la, ha, terms):
                pick = "secondary"
                swaps.append((am[i1].start(), am[i2 - 1].end(), tb))
        conflicts.append({"primary": _join(pa), "secondary": _join(sb), "pick": pick})
    text = pt
    for s, e, rep in reversed(swaps):
        text = text[:s] + rep + text[e:]
    agree = same / max(len(a), len(b))
    return {"text": text, "agree": round(agree, 3), "conflicts": conflicts}
