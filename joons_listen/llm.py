"""로컬 LLM(Ollama) 연결: 한국어 번역과 회의록. 로컬 주소로만 보낸다(외부 클라우드 API 사용 안 함)."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from . import config


class LLMError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _is_local(url: str) -> bool:
    return bool(re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?(/|$)", url))


def _req(path: str, body: Optional[Dict] = None, timeout: int = 20) -> Dict:
    url = config.LLM_URL.rstrip("/") + path
    if not _is_local(url):
        raise LLMError("LLM_UNAVAILABLE", "meeting text is sent to a local LLM only (JOONS_LISTEN_LLM_URL must be a loopback address)")
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        # 시스템·VPN 프록시를 타지 않도록 프록시 없이 직접 연결한다(회의 내용이 프록시를 거치지 않게)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise LLMError("LLM_UNAVAILABLE", f"cannot reach the local LLM (Ollama): {e.reason}")
    except Exception as e:  # noqa: BLE001
        raise LLMError("LLM_UNAVAILABLE", f"local LLM call failed: {str(e)[:160]}")


def available() -> Optional[str]:
    """쓸 수 있는 모델 이름(설정한 모델 또는 설치된 첫 모델). 없으면 None."""
    try:
        tags = _req("/api/tags", timeout=4).get("models", [])
    except LLMError:
        return None
    names = [m.get("name", "") for m in tags]
    if config.LLM_MODEL in names:
        return config.LLM_MODEL
    base = config.LLM_MODEL.split(":")[0]
    for n in names:
        if n.split(":")[0] == base:
            return n
    return None


def unload(model: Optional[str] = None) -> None:
    """LLM 이 쥔 GPU 메모리를 바로 돌려준다(8GB GPU 에서 다음 ASR 모델이 올라가야 하므로 번역·회의록이 끝나면 반드시 호출)."""
    m = model or available()
    if not m:
        return
    try:
        _req("/api/generate", {"model": m, "keep_alive": 0}, 20)
    except LLMError:
        pass


def chat(messages: List[Dict], model: str, json_mode: bool = False, timeout: Optional[int] = None) -> str:
    body: Dict = {"model": model, "messages": messages, "stream": False, "think": False, "keep_alive": "2m", "options": {"temperature": 0.1, "num_ctx": 8192}}
    if json_mode:
        body["format"] = "json"
    r = _req("/api/chat", body, timeout or config.LLM_TIMEOUT)
    return (r.get("message") or {}).get("content", "")


def _json_loose(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def translate_lines(lines: List[str], model: str, batch: int = 12) -> List[Optional[str]]:
    """중국어·영어 등 → 한국어. 길이가 맞지 않거나 실패한 항목은 None(원문 유지)."""
    out: List[Optional[str]] = [None] * len(lines)
    for b in range(0, len(lines), batch):
        chunk = lines[b : b + batch]
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
        sys = "당신은 회의 통역사입니다. 입력 문장을 자연스러운 한국어로 번역합니다. 고유명사·회사명·제품 코드·숫자·단위는 그대로 두고 임의로 바꾸거나 덧붙이지 않습니다. 중국어 회사·브랜드 이름은 아는 한 원어(한자)나 영문 표기를 그대로 두고 뜻풀이로 바꾸지 않습니다(예: 大疆은 DJI). 중국어의 元·块·块钱은 한국 원이 아니라 위안(RMB)으로, 万은 만, 亿는 억으로 옮기고 환율 환산은 하지 않습니다. 이미 한국어면 그대로 둡니다. 반드시 JSON {\"t\":[번역1,번역2,...]} 만 출력하고 항목 수는 입력과 같아야 합니다."
        try:
            txt = chat([{"role": "system", "content": sys}, {"role": "user", "content": numbered}], model, json_mode=True)
            arr = _json_loose(txt).get("t")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(arr, list) and len(arr) == len(chunk):
            for i, v in enumerate(arr):
                out[b + i] = str(v).strip() or None
    return out


_KRW_AFTER_NUM = re.compile(r"(?<=[0-9만억천백십])(\s*)원(?:(?=[에이은는을를의도으까부씩짜])|(?![가-힣]))")


def fix_cny(ko: str, src_lang: Optional[str]) -> str:
    """중국어 원문을 번역한 문장에서 숫자 뒤의 '원'은 한국 원이 아니라 元·块(위안)이다. 모델이 규칙을 놓치는 경우를 사후에 바로잡는다."""
    if src_lang != "zh" or not ko:
        return ko
    return _KRW_AFTER_NUM.sub(lambda m: m.group(1) + "위안", ko)


def make_minutes(transcript_lines: List[str], facts: List[str], title: str, model: str) -> Dict:
    """전사(화자·시각 포함)로 회의록 JSON 을 만든다. 긴 회의는 조각 요약 후 합친다."""
    def ask(text: str, final: bool) -> Dict:
        sys = (
            "당신은 회의록 작성자입니다. 주어진 전사에서 사실만 뽑아 한국어로 정리합니다. 전사에 없는 내용·숫자·약속을 지어내지 않습니다. 숫자·날짜·가격은 전사 그대로 옮기고, 불확실하면 '(확인 필요)'를 붙입니다. 전사에 연도가 없으면 연도를 붙이지 않고 말한 그대로('11월 15일', '다음 달 15일') 적으며, 기한을 모르면 빈 문자열로 둡니다. "
            "JSON 형식: {\"title\":str,\"summary\":str(5문장 이내),\"decisions\":[str],\"actions\":[{\"who\":str,\"what\":str,\"due\":str}],\"open_questions\":[str]} 만 출력합니다."
        )
        user = ("다음은 회의 전사의 일부입니다. 이 부분의 요약·결정·후속 조치를 정리하세요.\n" if not final else "다음은 구간별 요약입니다. 하나의 회의록으로 합치세요(중복 제거).\n") + text
        return _json_loose(chat([{"role": "system", "content": sys}, {"role": "user", "content": user}], model, json_mode=True))

    text = "\n".join(transcript_lines)
    limit = 5500
    if len(text) <= limit:
        res = ask(text, final=False)
    else:
        parts, cur = [], ""
        for ln in transcript_lines:
            if len(cur) + len(ln) > limit and cur:
                parts.append(cur)
                cur = ""
            cur += ln + "\n"
        if cur:
            parts.append(cur)
        partials = [json.dumps(ask(p, final=False), ensure_ascii=False) for p in parts]
        # 조각 요약이 많아 합칠 입력이 한도를 넘으면 묶음 단위로 먼저 합친다(앞부분이 잘려 나가지 않도록)
        while len("\n".join(partials)) > limit and len(partials) > 1:
            cur, nxt = "", []
            for p_ in partials:
                if cur and len(cur) + len(p_) > limit:
                    nxt.append(json.dumps(ask(cur, final=True), ensure_ascii=False))
                    cur = ""
                cur += p_ + "\n"
            if cur:
                nxt.append(json.dumps(ask(cur, final=True), ensure_ascii=False))
            if len(nxt) >= len(partials):
                break
            partials = nxt
        res = ask("\n".join(partials), final=True)
    res.setdefault("title", title)
    for k in ("decisions", "open_questions"):
        res[k] = [str(x) for x in (res.get(k) or []) if x]
    acts = []
    for a in res.get("actions") or []:
        if isinstance(a, dict):
            acts.append({"who": str(a.get("who", "")), "what": str(a.get("what", "")), "due": str(a.get("due", ""))})
    res["actions"] = acts
    res["summary"] = str(res.get("summary", ""))
    res["llm"] = model
    return res
