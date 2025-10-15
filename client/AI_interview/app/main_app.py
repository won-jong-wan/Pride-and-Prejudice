import os, time, tempfile
from pathlib import Path
import requests
import streamlit as st

from app.upload_section import render_upload_section
from core.storage import get_save_dir, safe_name, save_bytes
from core.remote_fetch import wait_until_ready
from core.analysis_pose import parse_posture_summary
from core.whisper import load_whisper, transcribe_file
from core.chains import build_feedback_chain
from core.analysis_audio import analyze_stability, get_stability_score

# ── 페이지/세션 기본 설정 ─────────────────────────────────────────────
st.set_page_config(page_title="AI 면접관", page_icon="🎤", layout="wide")

ss = st.session_state
ss.setdefault("stopped", False)
ss.setdefault("recording", False)
ss.setdefault("auto_mp4", True)
ss.setdefault("mp4_manual_only", False)     # MP4 수동 받기 
ss.setdefault("auto_saved_once", False)
ss.setdefault("session_id", "default_session")

# ── 외부 서버 주소(라즈베리파이 Flask) ───────────────────────────────
SERVER_URL = os.environ.get("PI_SERVER_URL", "http://10.10.14.80:5000")

# ── 저장 경로 준비 ────────────────────────────────────────────────────
BASE_DIR = Path("downloaded")
SAVE_DIR = get_save_dir(BASE_DIR, ss["session_id"])

# ── 모델/체인(세션당 1회 로드) ────────────────────────────────────────
if "whisper_model" not in ss:
    ss["whisper_model"] = load_whisper("small")  # small/medium/large

def load_google_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return ""

GOOGLE_API_KEY = load_google_key()
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되어 있지 않습니다. 환경변수 또는 .streamlit/secrets.toml로 설정해주세요.")
    st.stop()

# --- 체인: 세션당 1회만 생성 ---
ss = st.session_state
if "feedback_chain" not in ss:
    ss["feedback_chain"] = build_feedback_chain(
        api_key=GOOGLE_API_KEY,
        model="gemini-2.5-flash",
        temperature=0.6,
    )

# ── UI 타이틀 ─────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF; font-size: 40px;'>🎤 AI 면접관</h1>
    <h3 style='text-align: center; color: #555;'>당신의 <b>답변 · 자세 · 표정</b>을 종합 분석합니다</h3>
    """,
    unsafe_allow_html=True
)
st.divider()

# ── 컨트롤 영역 ───────────────────────────────────────────────────────
st.subheader("🎙️ 면접을 시작하시겠습니까?")
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ 면접 시작", use_container_width=True):
        try:
            r = requests.get(f"{SERVER_URL}/command/start_record", timeout=5)
            r.raise_for_status()
            ss["recording"] = True
            ss["stopped"] = False
            ss["auto_saved_once"] = False
            st.success("면접이 시작되었습니다.")
        except requests.exceptions.RequestException as e:
            st.error(f"면접 시작 실패: {e}")

with col2:
    if st.button("⏹️ 면접 종료", use_container_width=True):
        try:
            r = requests.get(f"{SERVER_URL}/command/stop_record", timeout=5)
            r.raise_for_status()
            ss["recording"] = False
            ss["stopped"] = True
            st.success("면접이 종료되었습니다.")
            st.rerun()  # 바로 자동저장 블록 실행
        except requests.exceptions.RequestException as e:
            st.error(f"면접 종료 실패: {e}")

# 상태 표시
st.markdown("🔴 **현재 면접 중입니다...**" if ss.get("recording") else "⚪ **대기 중**")

# MP4 수동 옵션
st.checkbox("MP4는 수동으로 받기", key="mp4_manual_only")

# ── 서버 파일 자동 저장(종료 후 1회) ──────────────────────────────────
FILES = {
    "mp4": {"url": f"{SERVER_URL}/download/mp4/video.mp4", "mime": "video/mp4",       "name": "video.mp4"},
    "wav": {"url": f"{SERVER_URL}/download/wav/audio.wav", "mime": "audio/wav",       "name": "audio.wav"},
    "xml": {"url": f"{SERVER_URL}/download/xml/log.xml",   "mime": "application/xml", "name": "log.xml"},
}
order = ["mp4", "wav", "xml"]
auto_all = ss["auto_mp4"]
auto_kinds = [k for k in order if not (k == "mp4" and ss["mp4_manual_only"])]

if ss.get("stopped") and auto_all and auto_kinds and not ss["auto_saved_once"]:
    ready = {}
    with st.spinner("파일 준비 확인 중..."):
        for kind in auto_kinds:
            info = FILES[kind]
            res = wait_until_ready(info["url"], max_wait_s=10, interval_s=0.5)
            if res and res.content:
                name = safe_name(Path(info["name"]).stem, ss["session_id"], Path(info["name"]).suffix)
                p = save_bytes(SAVE_DIR, name, res.content)
                ready[kind] = p

    if ready:
        ss["auto_saved_once"] = True
        st.success("자동 저장 완료! (로컬 디스크)")
        for k, p in ready.items():
            st.write(f"✅ `{k}` 저장: `{p.resolve()}`")
    else:
        st.info("파일이 아직 준비되지 않았습니다.")

# MP4 수동 다운로드(수동 모드이거나 auto_all=False인 경우)
if st.session_state.get("mp4_manual_only", True) or not auto_all:
    if st.button("⬇️ MP4 받기", use_container_width=True):
        info = FILES["mp4"]
        res = wait_until_ready(info["url"], max_wait_s=10, interval_s=0.5)
        if res and res.content:
            name = safe_name(Path(info["name"]).stem, ss["session_id"], Path(info["name"]).suffix)
            p = save_bytes(SAVE_DIR, name, res.content)
            st.success("MP4 다운로드 준비 완료!")
            st.download_button("파일 저장(브라우저)", data=res.content, file_name=name, mime=info["mime"], use_container_width=True)
            st.caption(f"로컬 저장: `{p.resolve()}`")
        else:
            st.warning("MP4가 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요.")

render_upload_section(
    SAVE_DIR=SAVE_DIR,          # Path 객체 (예: downloaded/<SESSION_ID>)
    ss=st.session_state,        # 세션 상태
    whisper_model=ss["whisper_model"],
    feedback_chain=ss["feedback_chain"],
)

st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>© 2025 AI Interview Project | Powered by Streamlit · Whisper · LangChain · Hailo</div>",
    unsafe_allow_html=True
)
