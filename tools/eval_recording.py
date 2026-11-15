"""실제 녹음 평가: 사람이 정리한 회의록의 핵심 항목(체크리스트)이 전사문에 나오는지(재현율)를 센다.
사용: python tools/eval_recording.py --checklist examples/checklist.example.json --text part1.txt part2.txt [--out result.json]
체크리스트 형식: [{"item": "설명", "any": ["정규식", "..."], "all": ["정규식"]?}] — any 중 하나라도, all 은 전부 나와야 인정.
정답 원고 대신 '회의록에 적힌 사실이 전사에 살아 있는가'를 보는 것이라 글자 오류율이 아니라 핵심 정보 재현율이다."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", "", t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    text = norm("\n".join(Path(p).read_text(encoding="utf-8") for p in a.text))
    items = json.loads(Path(a.checklist).read_text(encoding="utf-8"))
    rows, hit = [], 0
    for it in items:
        anys = it.get("any", [])
        alls = it.get("all", [])
        if not anys and not alls:  # 조건이 없는 항목이 무조건 '찾음'으로 세어지는 것을 막는다
            raise SystemExit(f"체크리스트 항목에 any/all 이 없습니다: {it.get('item')}")
        ok_any = (not anys) or any(re.search(norm(p), text, re.I) for p in anys)
        ok_all = all(re.search(norm(p), text, re.I) for p in alls)
        ok = ok_any and ok_all
        hit += ok
        rows.append({"item": it["item"], "found": ok})
    res = {"total": len(items), "found": hit, "recall": round(hit / max(1, len(items)), 3), "rows": rows}
    for r in rows:
        print(("O " if r["found"] else "X ") + r["item"])
    print(f"재현율 {hit}/{len(items)} = {res['recall']:.0%}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
