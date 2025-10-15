import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Piece:
    text: str; t0: float; t1: float
    meta: Dict[str, Any] = field(default_factory=dict)

class RealtimeBuffer:
    def __init__(self, silence_gap=0.9):
        self.buf: List[Piece] = []; self.silence_gap = silence_gap
    def add_text(self, text: str, t0: float, t1: float, meta=None):
        text = (text or "").strip()
        if text: self.buf.append(Piece(text, t0, t1, meta or {}))
    def add_event(self, ev: Dict[str, Any]):
        if self.buf: self.buf[-1].meta.setdefault("events", []).append(ev)
    def ready(self) -> bool:
        if not self.buf: return False
        now = time.time(); last = self.buf[-1]
        return (now - last.t1) >= self.silence_gap or last.text.endswith((".", "?", "!", "…"))
    def flush(self) -> Optional[Dict[str, Any]]:
        if not self.buf: return None
        utt = " ".join(p.text for p in self.buf).strip()
        t0, t1 = self.buf[0].t0, self.buf[-1].t1
        evs = []; [evs.extend(p.meta.get("events", [])) for p in self.buf]
        self.buf.clear()
        return {"text": utt, "t0": t0, "t1": t1, "events": evs}

def summarize_events(events: list) -> str:
    return "없음" if not events else "; ".join(f"{e.get('type')}:{e.get('score')}" for e in events)
