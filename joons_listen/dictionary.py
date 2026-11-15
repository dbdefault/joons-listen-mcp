"""사용자 사전: 회사명·고객명·인명·제품명. ASR 힌트(hotwords/context)와 사후 교정(from→to)에 쓴다.

파일 형식(JSON): {"terms":[{"term":"Acme(艾克米)","type":"company","aliases":["艾克","艾克米","에이크미","Acme"]}], "replace":[{"from":"ㅇㅇ","to":"ㅇㅇ"}]}
- aliases 는 같은 대상을 가리키는 다른 표기. 전사문에서 alias 가 나오면 `term` 으로 통일하지 않고(오교정 위험) 원문을 보존하며 facts 에 'name' 후보로만 올린다.
- replace 는 확실한 오인식만 넣는다(명시적으로 적은 것만 교정한다).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import config


def load(path: Optional[Path] = None) -> Dict:
    p = Path(path) if path else config.DICT_PATH
    if not p.exists():
        return {"terms": [], "replace": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"terms": [], "replace": []}
    data.setdefault("terms", [])
    data.setdefault("replace", [])
    return data


def save(data: Dict, path: Optional[Path] = None) -> None:
    p = Path(path) if path else config.DICT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def hint_words(data: Dict, limit: int = 60) -> List[str]:
    """ASR 힌트로 줄 낱말(고유명·별칭). 너무 많으면 앞에서 limit 개."""
    out: List[str] = []
    for t in data.get("terms", []):
        for w in [t.get("term", "")] + list(t.get("aliases", [])):
            w = str(w).strip()
            if w and w not in out:
                out.append(w)
    return out[:limit]


def is_hint_echo(text: str, hints: Iterable[str], min_hits: int = 4) -> bool:
    """잡음 구간에서 ASR 이 힌트(사전 낱말 목록)를 그대로 받아쓴 경우. 서로 다른 힌트가 min_hits 개 이상 나오고
    글자의 절반 이상이 힌트로 채워지면 환각으로 본다(실측: 원거리 녹음 잡음 구간에서 고유명 목록이 전사로 나옴)."""
    t = str(text or "")
    if len(t) < 20:
        return False
    hs = sorted({str(h).strip() for h in hints if str(h).strip()}, key=len, reverse=True)
    hits, rest = 0, t
    for h in hs:
        if h in rest:
            hits += 1
            rest = rest.replace(h, "")
    if hits < min_hits:
        return False
    filler = sum(1 for c in rest if not c.isalnum())
    return (len(rest) - filler) <= 0.5 * len(t)


def merge_job_replace(data: Dict, extra: Optional[Iterable[Dict]]) -> List[Dict]:
    rep = list(data.get("replace", []))
    for r in extra or []:
        if isinstance(r, dict) and r.get("from"):
            rep.append({"from": str(r["from"]), "to": str(r.get("to", ""))})
    return rep


def apply_replace(text: str, rules: Iterable[Dict]) -> str:
    for r in rules:
        src = r.get("from", "")
        if not src:
            continue
        to = str(r.get("to", ""))
        text = re.sub(re.escape(str(src)), lambda m, _to=to: _to, text, flags=re.IGNORECASE)  # to 안의 \1 같은 백슬래시를 정규식 치환 문법으로 해석하지 않는다
    return text


def name_hits(text: str, data: Dict) -> List[str]:
    """전사문에서 사전의 별칭이 나온 항목의 대표 표기를 돌려준다(사실 후보용)."""
    hits: List[str] = []
    low = text.lower()
    for t in data.get("terms", []):
        words = [t.get("term", "")] + list(t.get("aliases", []))
        if any(w and w.lower() in low for w in words):
            hits.append(t.get("term", ""))
    return hits


_FOLDER_RE = re.compile(r"^(?:\d+[_\-\s]*)?(?P<name>[A-Za-z][A-Za-z0-9 &\-]{1,30})\s*[(（]?(?P<zh>[一-鿿]{2,10})[)）]?")


def build_from_folder_names(names: Iterable[str]) -> Dict:
    """폴더 이름(예: '01_Acme艾克米', '03_Globex格罗贝')에서 사전 초안을 만든다. 결과는 사람이 검토한다."""
    terms: List[Dict] = []
    seen = set()
    for n in names:
        m = _FOLDER_RE.match(n.strip())
        if not m:
            continue
        en, zh = m.group("name").strip(), m.group("zh")
        key = en.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append({"term": f"{en}({zh})", "type": "company", "aliases": sorted({en, zh})})
    return {"terms": terms, "replace": []}
