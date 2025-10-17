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