# core/recording_io.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import requests, time, re

BASE_DIR = Path("data/records")

def get_save_dir(session_id: str) -> Path:
    p = BASE_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def wait_until_ready(url: str, max_wait_s: int = 10, interval_s: float = 0.5) -> Optional[requests.Response]:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=(3.05, 8))
            if r.ok and r.content:
                return r
        except Exception:
            pass
        time.sleep(interval_s)
    return None

def safe_name(stem: str, session_id: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    sid  = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id or "sess")
    return f"{stem}_{sid}{suffix}"

def save_bytes(dirpath: Path, filename: str, content: bytes) -> Path:
    p = dirpath / filename
    p.write_bytes(content)
    return p

def save_assets_after_stop(
    server_url: str,
    session_id: str,
    kinds: list[str] = ["mp4", "wav", "xml"],
) -> Dict[str, Path]:
    """서버의 고정 경로에서 mp4/wav/xml을 받아 세션 폴더에 저장하고, 저장된 경로를 반환."""
    files = {
        "mp4": {"url": f"{server_url}/download/mp4/video.mp4", "name": "video.mp4"},
        "wav": {"url": f"{server_url}/download/wav/audio.wav", "name": "audio.wav"},
        "xml": {"url": f"{server_url}/download/xml/log.xml",   "name": "log.xml"},
        #"mp4": {"url": f"{server_url}/download/mp4/video_ai.mp4", "name": "video_ai.mp4"},
    }
    out: Dict[str, Path] = {}
    save_dir = get_save_dir(session_id)
    for k in kinds:
        info = files.get(k)
        if not info:
            continue
        res = wait_until_ready(info["url"], max_wait_s=10, interval_s=0.5)
        if res and res.content:
            fname = safe_name(Path(info["name"]).stem, session_id, Path(info["name"]).suffix)
            out[k] = save_bytes(save_dir, fname, res.content)
    return out
