"""화자 분리(비게이트 구성): silero-vad 구간 → 짧은 창 → ECAPA 임베딩 → 코사인 군집 → 화자별 발언(turn).

pyannote 파이프라인은 Hugging Face 로그인·약관 동의가 필요한 게이트 모델이라 쓰지 않는다. 이 방식은 로그인 없이 동작하며,
한중영 혼용·겹치는 발언·아주 짧은 맞장구에서는 한계가 있다(결과의 speakers 와 flags 로 확인).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .audio import SR, slice_audio

_classifier = None


@dataclass
class Turn:
    start: float
    end: float
    speaker: str  # "S1", "S2", ...

    def to_dict(self) -> Dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "speaker": self.speaker}


# ---------- 순수 함수(시험 가능) ----------
def make_windows(segments: Sequence[Tuple[float, float]], win: float = config.DIAR_WINDOW_SEC, min_len: float = 0.8) -> List[Tuple[float, float, int]]:
    """VAD 구간을 임베딩용 창으로 자른다. 반환 (시작, 끝, 구간 번호). 너무 짧은 구간은 한 창으로 둔다."""
    out: List[Tuple[float, float, int]] = []
    for idx, (s, e) in enumerate(segments):
        length = e - s
        if length <= win * 1.5:
            out.append((s, e, idx))
            continue
        t = s
        while t < e - 1e-6:
            u = min(t + win, e)
            if e - u < min_len:  # 꼬리가 너무 짧으면 앞 창에 붙인다
                u = e
            out.append((t, u, idx))
            t = u
    return out


def cluster_embeddings(emb: np.ndarray, num_speakers: Optional[int] = None, threshold: float = config.DIAR_THRESHOLD, max_speakers: int = 10) -> np.ndarray:
    """정규화된 임베딩 (N, D) → 화자 번호 배열(0..K-1)."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    if num_speakers and num_speakers >= 1:
        k = min(int(num_speakers), n)
        if k == 1:
            return np.zeros(n, dtype=int)
        return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(e)
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average").fit_predict(e)
    k = len(set(labels.tolist()))
    if k > max_speakers:
        labels = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine", linkage="average").fit_predict(e)
    return labels


def merge_tiny_speakers(emb: np.ndarray, labels: np.ndarray, durations: np.ndarray, min_sec: float = 5.0, min_ratio: float = 0.02) -> np.ndarray:
    """총 발언이 아주 짧은 군집은 가장 가까운 큰 군집으로 합친다(잡음·기침이 화자가 되는 것을 막는다)."""
    labels = labels.copy()
    total = float(durations.sum()) or 1.0
    ids = sorted(set(labels.tolist()))
    if len(ids) <= 1:
        return labels
    sums = {i: float(durations[labels == i].sum()) for i in ids}
    big = [i for i in ids if sums[i] >= max(min_sec, min_ratio * total)]
    if not big:  # 전부 작으면 가장 긴 하나를 기준으로
        big = [max(ids, key=lambda i: sums[i])]
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cents = {i: e[labels == i].mean(axis=0) for i in big}
    for i in ids:
        if i in big:
            continue
        c = e[labels == i].mean(axis=0)
        best = max(big, key=lambda b: float(np.dot(c, cents[b]) / (np.linalg.norm(c) * np.linalg.norm(cents[b]) + 1e-9)))
        labels[labels == i] = best
    return labels


def relabel_by_first_appearance(starts: Sequence[float], labels: np.ndarray) -> List[str]:
    order: Dict[int, int] = {}
    for idx in np.argsort(np.asarray(starts)):
        lab = int(labels[idx])
        if lab not in order:
            order[lab] = len(order) + 1
    return [f"S{order[int(l)]}" for l in labels]


def segments_to_turns(segments: Sequence[Tuple[float, float]], seg_speakers: Sequence[str], gap: float = config.TURN_GAP_SEC, max_len: float = config.TURN_MAX_SEC) -> List[Turn]:
    """같은 화자의 가까운 구간을 이어 붙여 발언 단위(Turn)로 만든다."""
    turns: List[Turn] = []
    for (s, e), spk in sorted(zip(segments, seg_speakers), key=lambda x: x[0][0]):
        if turns and turns[-1].speaker == spk and s - turns[-1].end <= gap and e - turns[-1].start <= max_len:
            turns[-1].end = max(turns[-1].end, e)
        else:
            turns.append(Turn(s, e, spk))
    return turns


def assign_segment_speakers(windows: Sequence[Tuple[float, float, int]], win_labels: Sequence[int], n_segments: int) -> List[int]:
    """창별 화자 → 구간별 화자(길이 가중 다수결)."""
    votes: List[Dict[int, float]] = [dict() for _ in range(n_segments)]
    for (s, e, idx), lab in zip(windows, win_labels):
        votes[idx][int(lab)] = votes[idx].get(int(lab), 0.0) + (e - s)
    return [max(v, key=lambda k: v[k]) if v else 0 for v in votes]


def windows_to_turns(windows: Sequence[Tuple[float, float, int]], win_labels: Sequence[int], gap: float = config.TURN_GAP_SEC, max_len: float = config.TURN_MAX_SEC, smooth: bool = True) -> List[Turn]:
    """창별 화자 → 발언 목록. 시간순으로 놓고, 양옆이 같은 화자인 고립된 짧은 창은 이웃 화자로 고친 뒤(잡음 방지),
    같은 화자의 가까운 창을 이어 붙인다. 화자 번호는 처음 나온 순서로 S1, S2…."""
    if not windows:
        return []
    order = sorted(range(len(windows)), key=lambda i: windows[i][0])
    ws = [windows[i] for i in order]
    lab = [int(win_labels[i]) for i in order]
    if smooth and len(ws) >= 3:
        for i in range(1, len(ws) - 1):
            adj = (ws[i][0] - ws[i - 1][1] <= gap) and (ws[i + 1][0] - ws[i][1] <= gap)
            if adj and lab[i - 1] == lab[i + 1] != lab[i] and (ws[i][1] - ws[i][0]) <= config.DIAR_WINDOW_SEC * 1.1:
                lab[i] = lab[i - 1]
    names = relabel_by_first_appearance([w[0] for w in ws], np.array(lab))
    turns: List[Turn] = []
    for (s, e, _), spk in zip(ws, names):
        if turns and turns[-1].speaker == spk and s - turns[-1].end <= gap and e - turns[-1].start <= max_len:
            turns[-1].end = max(turns[-1].end, e)
        else:
            turns.append(Turn(s, e, spk))
    return turns


# ---------- 모델 의존 부분 ----------
def _get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    import torch
    from huggingface_hub import snapshot_download

    local = config.MODELS_DIR / "ecapa"
    if not (local / "hyperparams.yaml").exists():
        snapshot_download(config.ECAPA_REPO, local_dir=str(local))
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:  # 옛 버전
        from speechbrain.pretrained import EncoderClassifier  # type: ignore
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {}
    try:
        from speechbrain.utils.fetching import LocalStrategy

        kw["local_strategy"] = LocalStrategy.COPY  # Windows 에서 심볼릭 링크 권한 오류를 피한다
    except Exception:
        pass
    _classifier = EncoderClassifier.from_hparams(source=str(local), savedir=str(local), run_opts={"device": dev}, **kw)
    return _classifier


def release() -> None:
    global _classifier
    _classifier = None
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def embed_windows(audio: np.ndarray, windows: Sequence[Tuple[float, float, int]], progress=None) -> np.ndarray:
    import torch

    clf = _get_classifier()
    embs: List[np.ndarray] = []
    total = max(1, len(windows))
    for i, (s, e, _) in enumerate(windows):
        chunk = slice_audio(audio, s, e)
        if len(chunk) < int(0.3 * SR):
            chunk = np.pad(chunk, (0, int(0.3 * SR) - len(chunk)))
        with torch.no_grad():
            v = clf.encode_batch(torch.from_numpy(chunk).unsqueeze(0)).squeeze().detach().cpu().numpy()
        embs.append(v.astype(np.float32))
        if progress and (i % 20 == 0 or i == total - 1):
            progress((i + 1) / total)
    return np.vstack(embs) if embs else np.zeros((0, 192), dtype=np.float32)


def diarize(audio: np.ndarray, segments: Sequence[Tuple[float, float]], num_speakers: Optional[int] = None, progress=None) -> Tuple[List[Turn], Dict]:
    """VAD 구간들 → 화자가 붙은 발언 목록과 요약 정보."""
    if not segments:
        return [], {"speakers": 0, "windows": 0}
    windows = make_windows(segments)
    emb = embed_windows(audio, windows, progress)
    labels = cluster_embeddings(emb, num_speakers)
    durs = np.array([e - s for s, e, _ in windows], dtype=float)
    if not num_speakers:
        labels = merge_tiny_speakers(emb, labels, durs)
    # 창 단위 화자로 발언을 자른다(긴 VAD 구간 안에서 말하는 사람이 바뀌어도 따라가도록).
    turns = windows_to_turns(windows, labels)
    return turns, {"speakers": len(set(t.speaker for t in turns)), "windows": len(windows)}
