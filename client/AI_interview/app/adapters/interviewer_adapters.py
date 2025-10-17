from __future__ import annotations
from typing import Tuple
import tempfile
from pathlib import Path
from functools import lru_cache
from core import whisper_run, chains  # <- 방금 이름 바꾼 파일을 임포트

# TTS는 보류(나중에 붙일 예정)
def my_tts_interviewer(text: str, speed: float = 0.95) -> Tuple[bytes, str]:
    raise NotImplementedError

@lru_cache(maxsize=1)
def _get_whisper_model():
    # 필요 시 사이즈 변경: "base", "small", "medium", "large"
    return whisper_run.load_whisper(model_size="small")

def my_stt_from_path(wav_path: str) -> str:
    """
    - 파일 경로를 그대로 Whisper에 전달해 STT 수행
    - 반환: result["text"]
    """
    model = _get_whisper_model()  # 네 파일에 이미 있는 헬퍼 그대로 사용
    result = whisper_run.transcribe_file(
        model,
        wav_path,               # ← 경로 직접 전달
        language="ko",
        task="transcribe",
        temperature=0.0,
    )
    return (result.get("text") or "").strip()

def my_feedback(question: str, answer: str, nonverbal: str | None = None) -> str:
    if not (answer and answer.strip()):
        return "답변이 비어 있어요. 한 문장으로 핵심 결론부터 말해보세요."
    latest_utt = (answer or "")[:800]                 # 방어적 슬라이스
    nv = (nonverbal or "특이사항 없음")[:300]
    try:
        chain = chains.get_feedback_chain()        # 캐시된 LLMChain
        return chain.run({"latest_utt": latest_utt, "nonverbal": nv})
    except Exception as e:
        return f"(임시 코칭) 첫 문장에 결론을 제시하고, 사례/수치를 1개만 보태 보세요. // 오류: {e}"
 