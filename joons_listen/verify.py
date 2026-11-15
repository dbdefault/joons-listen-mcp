"""가격·수량·날짜·제품 코드·고유명 후보를 뽑고, 불확실한 구간에 표시를 붙인다. 모든 사실은 '검증 필요'로 둔다(자동 확정 금지)."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

NUM = r"\d(?:[\d,\.]*\d)?"  # 끝이 숫자여야 한다(문장 끝 쉼표·마침표를 숫자로 먹지 않는다)
# 중국어 숫자(一万二、两千、十四 등). 아라비아 숫자와 함께 가격·수량 후보로 뽑는다.
CNUM = r"[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?"
# 승수: 숫자 뒤에 붙는 万·千·亿·만·천·억(예: 5万吨, 2만톤, 3억원)
MULT = r"(?:[万千亿만천억]|million|thousand)?"
RE_PRICE = re.compile(
    rf"(?:[¥￥$€]\s*{NUM}(?:\s*[万千kKmM])?|(?:USD|CNY|RMB|EUR|人民币)\s*{NUM}"
    rf"|{NUM}\s*{MULT}\s*(?:万元|万块|元|块钱|块|RMB|rmb|CNY|cny|USD|usd|yuan|美元|美金|刀|달러|불|원|위안)(?:/\s*(?:吨|톤|ton|kg|公斤|KG|MT|pc|pcs))?"
    rf"|{CNUM}\s*(?:万元|万块|块钱|块|元|美元|美金|刀)(?:/\s*(?:吨|公斤|KG|MT))?"
    rf"|(?:{NUM}|{CNUM})\s*(?:元|块|美元|美金)?\s*[一每]\s*(?:吨|公斤|斤|升)(?![吨公斤升]))"
)
RE_QTY = re.compile(
    rf"(?:{NUM}(?:\s*[-~～]\s*{NUM})?\s*{MULT}|{CNUM})\s*(?:吨|톤|tons?|Tons?|MT|mt|kg|KG|公斤|千克|㎏|个(?!月)|개(?!월)|EA|ea|袋|포대|箱|박스|柜|컨테이너|FCL|fcl|方|㎥|m3|米|미터|m\b|条|台|套|线|라인|대|pallets?|cartons?|units?|pcs)"
)
RE_DATE = re.compile(
    r"(?:(?<!\d)\d{4}\s*[年./\-]\s*\d{1,2}\s*[月./\-]\s*\d{1,2}\s*[日号]?(?!\d)|\d{1,2}\s*月\s*\d{1,2}\s*[日号]|\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}\s*년\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"
    r"|(?<![\d/])(?:1[0-2]|0?[1-9])/(?:3[01]|[12]\d|0?[1-9])(?![\d/])|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?(?!\d)(?:,?\s*\d{4})?|\d{4})|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?(?:,?\s*\d{4})?|"
    r"[一二三四五六七八九十]{1,3}\s*[号日]|"
    r"(?:下周|下个月|本周|本月|明天|后天|月底|月初|이번\s*주|다음\s*주|다음\s*달|내일|모레|월말|월초)[一二三四五六日天]?)"
)
# 제품 코드: \b 는 한글·한자도 단어 문자로 봐서 "型号PP-1100N", "PP-1100N입니다"처럼 붙어 나오면 놓치므로 ASCII 영숫자 기준 경계를 쓴다.
RE_CODE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,6}[- ]?\d{2,10}[A-Z]?|\d{2,4}[A-Z]{1,3}\d{0,3})(?![A-Za-z0-9])")

# '一个', '一块', '一套' 처럼 낱개를 뜻하는 흔한 말은 가격·수량 후보에서 뺀다(잡음).
_TRIVIAL = re.compile(r"[一二两]\s*[个条台套线块方袋箱]")

_STOP_CODES = {"MT", "KG", "RMB", "USD", "CNY", "EA", "FCL", "M3"}


def extract_facts(text: str, segment: int) -> List[Dict]:
    """한 문장에서 사실 후보를 뽑는다. type: price|quantity|date|code."""
    facts: List[Dict] = []
    seen = set()

    def add(kind: str, m: re.Match) -> None:
        s = m.group(0).strip()
        key = (kind, s)
        if not s or key in seen:
            return
        seen.add(key)
        facts.append({"type": kind, "text": s, "segment": segment, "verify": True})

    for m in RE_PRICE.finditer(text):
        if _TRIVIAL.fullmatch(m.group(0).strip()):
            continue
        add("price", m)
    for m in RE_QTY.finditer(text):
        if _TRIVIAL.fullmatch(m.group(0).strip()):
            continue
        if ("price", m.group(0).strip()) in seen:  # '两万一吨' 같은 구어 단가는 수량으로 중복 표시하지 않는다
            continue
        add("quantity", m)
    for m in RE_DATE.finditer(text):
        add("date", m)
    for m in RE_CODE.finditer(text):
        raw = m.group(0).replace(" ", "").replace("-", "").upper()
        if raw in _STOP_CODES or re.sub(r"\d+[A-Z]?$", "", raw) in _STOP_CODES:
            continue
        add("code", m)
    return facts


def _repeated(text: str) -> bool:
    """같은 글자·같은 짧은 구절이 비정상적으로 반복되면(환각) True."""
    t = re.sub(r"\s+", "", text)
    if len(t) < 12:
        return False
    for n in (1, 2, 3, 4):
        grams = [t[i : i + n] for i in range(0, len(t) - n + 1, n)]
        if grams and max(grams.count(g) for g in set(grams)) / len(grams) > 0.6 and len(grams) >= 6:
            return True
    return False


def flag_segment(text: str, duration: float, confidence: float | None, low_conf: float = 0.45) -> List[str]:
    flags: List[str] = []
    stripped = text.strip()
    if not stripped:
        flags.append("empty")
        return flags
    if confidence is not None and confidence < low_conf:
        flags.append("uncertain")
    cps = len(re.sub(r"\s+", "", stripped)) / max(duration, 0.1)
    if duration >= 2.0 and (cps < 0.6 or cps > 14):
        flags.append("uncertain")
    if _repeated(stripped):
        flags.append("uncertain")
        flags.append("repeat")
    return sorted(set(flags))


def analyze(segments: List[Dict]) -> Dict:
    """segments(text, start, end, confidence) 에 flags 를 붙이고 facts·uncertain 목록을 만든다."""
    facts: List[Dict] = []
    uncertain: List[int] = []
    for seg in segments:
        i = seg["i"]
        text = seg.get("text", "")
        fl = flag_segment(text, seg["end"] - seg["start"], seg.get("confidence"))
        fs = extract_facts(text, i)
        for f in fs:
            kind = f["type"]
            if kind not in fl:
                fl.append(kind)
            if kind in ("price", "quantity", "date", "code") and "uncertain" not in fl and seg.get("confidence") is not None and seg["confidence"] < 0.7:
                fl.append("check_" + kind)
        seg["flags"] = sorted(set(fl))
        facts.extend(fs)
        if "uncertain" in seg["flags"] or any(x.startswith("check_") for x in seg["flags"]):
            uncertain.append(i)
    return {"facts": facts, "uncertain": sorted(set(uncertain))}
