from pathlib import Path
import time

def get_save_dir(base: Path, session_id: str) -> Path:
    p = base / _safe_session(session_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_session(raw: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(raw))

def safe_name(stem: str, session_id: str, suffix: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{_safe_session(session_id)}_{ts}{suffix}"

def save_bytes(save_dir: Path, name: str, data: bytes) -> Path:
    path = save_dir / name
    path.write_bytes(data)
    return path
