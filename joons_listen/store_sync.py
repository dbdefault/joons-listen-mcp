"""결과 미러링: 끝난 작업의 결과(원본 오디오 제외)를 zip 으로 묶어 미러 폴더(JOONS_LISTEN_MIRROR_DIR)에 복사한다.
미러 폴더가 잠시 없어도(네트워크 드라이브 등) 전사는 그대로 끝나고 mirror=pending 으로 남아 나중에 다시 복사한다."""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Tuple

from . import config

SKIP_NAMES = {"audio.wav"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(job_id: str, job_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    """zip(결과 파일 + manifest.json)과 완료 표식(.ok, zip 의 SHA256 한 줄)을 만든다. 표식이 있어야 받은 것으로 친다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(job_dir.iterdir()) if p.is_file() and p.name not in SKIP_NAMES and not p.name.endswith(".tmp")]
    manifest: Dict = {"job_id": job_id, "version": config.VERSION, "files": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in files}}
    zp = out_dir / f"{job_id}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    okp = out_dir / f"{job_id}.ok"
    okp.write_text(_sha(zp) + "\n", encoding="ascii")
    return zp, okp


def push(job_id: str, job_dir: Path, work_dir: Path) -> Tuple[bool, str]:
    """미러 폴더로 복사한다. zip 먼저, 표식은 마지막(받는 쪽이 표식을 보고서야 가져가므로 반쯤 온 파일은 무시된다)."""
    if config.MIRROR_MODE == "off":
        return True, "off"
    zp, okp = build_bundle(job_id, job_dir, work_dir / "sync")
    dest = Path(config.MIRROR_DIR)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for f in (zp, okp):
            shutil.copy2(f, dest / (f.name + ".part"))
            (dest / (f.name + ".part")).replace(dest / f.name)
    except OSError as e:
        return False, f"mirror folder copy failed: {str(e)[:200]}"
    for f in (zp, okp):  # 복사한 뒤 전송용 사본은 지운다(정본은 jobs 폴더)
        try:
            f.unlink()
        except OSError:
            pass
    return True, "ok"
