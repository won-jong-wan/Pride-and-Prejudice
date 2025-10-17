from __future__ import annotations
import streamlit as st
import requests, tempfile, base64, shutil,time
import os, wave
import numpy as np, math
import time
from pathlib import Path
from typing import Callable, Iterable, Tuple
from core.analysis_audio import analyze_stability, get_stability_score
from core.analysis_pose import parse_posture_summary
from adapters.interviewer_adapters import my_stt_from_path as stt_fn
from core.recording_io import save_assets_after_stop
from core.chains import get_prompt,call_llm

SHOW_PER_ANSWER_METRICS = False  # 답변별 지표는 숨김
SHOW_FINAL_METRICS      = True   # 총평에서만 지표 표시

# 넘파이 스칼라 유틸 
def _to_native(x):
    if isinstance(x, np.generic):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
        return v
    if isinstance(x, dict):  return {k:_to_native(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_to_native(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    return x

#path 방식 
def is_valid_wav(path: str) -> tuple[bool, str]:
    try:
        if not os.path.exists(path): return False, "파일 없음"
        if os.path.getsize(path) < 1024: return False, "파일 너무 작음"
        with wave.open(path, "rb") as w:
            if w.getnframes() <= 0: return False, "프레임 0"
        return True, "OK"
    except wave.Error as e:
        return False, f"wave 에러: {e}"
    except Exception as e:
        return False, f"기타 에러: {e}"
    
def download_wav_direct(server_url: str, max_wait_s=20, interval_s=0.5) -> Path:
    url = f"{server_url}/download/wav/audio.wav"
    deadline = time.time() + max_wait_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == 200 and "audio" in (r.headers.get("Content-Type", "").lower()):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                    shutil.copyfileobj(r.raw, tf)
                    return Path(tf.name)
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    raise RuntimeError(f"WAV 준비 실패: {last_err or 'timeout'}")
"""def _save_resp_to_tmp(resp: requests.Response, server_url: str | None = None) -> Path:
    
    stop_record 응답이
    1) audio/* 바이트
    2) JSON(base64: audio_b64/file_b64)
    3) JSON(URL/경로: file_url/server_path/file_id)
    모두 올 때 임시 wav 파일로 저장.
    server_url은 file_url이 상대경로일 때만 필요.
    
    ctype = resp.headers.get("Content-Type", "")

    # 1) audio/* 직접 전송
    if resp.status_code == 200 and "audio" in ctype:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            shutil.copyfileobj(resp.raw, tf)
            return Path(tf.name)

    # 그 외는 JSON 처리
    data = resp.json()

    # 2) base64
    b64 = data.get("audio_b64") or data.get("file_b64")
    if b64:
        raw = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tf.write(raw)
            return Path(tf.name)

    # 3) URL/경로
    file_url    = data.get("file_url")
    server_path = data.get("server_path") or data.get("path")
    file_id     = data.get("file_id")

    def _abs(u: str) -> str:
        if u.startswith("http"):
            return u
        if not server_url:
            raise RuntimeError("상대 file_url을 받았지만 server_url이 없습니다.")
        return f"{server_url}{u}"

    if file_url:
        r2 = requests.get(_abs(file_url), stream=True, timeout=60)
        r2.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            shutil.copyfileobj(r2.raw, tf)
            return Path(tf.name)

    if server_path:
        if not server_url:
            raise RuntimeError("server_path를 받았지만 server_url이 없습니다.")
        r2 = requests.get(f"{server_url}/download", params={"path": server_path}, stream=True, timeout=60)
        r2.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            shutil.copyfileobj(r2.raw, tf)
            return Path(tf.name)

    if file_id:
        if not server_url:
            raise RuntimeError("file_id를 받았지만 server_url이 없습니다.")
        r2 = requests.get(f"{server_url}/download", params={"file_id": file_id}, stream=True, timeout=60)
        r2.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            shutil.copyfileobj(r2.raw, tf)
            return Path(tf.name)

    raise RuntimeError(f"지원하지 않는 stop_record 응답 형식: keys={list(data.keys())}")"""

def resolve_posture_xml_for(wav_path: str) -> str | None:
    p = Path(wav_path); x = p.with_suffix(".xml")
    return str(x) if x.exists() else None

def render_interviewer_panel(
    server_url: str,
    tts_interviewer: Callable[[str, float], Tuple[bytes, str]],
    stt_fn: Callable[[str], str],
    feedback_fn: Callable[[str, str], str],
    questions: Iterable[str] = (
        "자기소개 부탁드립니다.",
        "가장 도전적이었던 프로젝트와 역할은?",
        "문제 해결 경험을 STAR 구조로 설명해 주세요.",
    ),
    tts_speed: float = 0.95,
) -> None:
    """
    면접관 모드 패널 하나로 끝.
    - server_url: 라즈베리 녹음 서버 기반 URL (예: http://10.0.0.5:8000)
    - tts_interviewer(question_text, speed)->(audio_bytes, mime)
    - stt_fn(wav_path)->text
    - feedback_fn(question_text, answer_text)->feedback_text
    - questions: 질문 리스트
    """
    ss = st.session_state
    # 세션키는 eva_* 접두사로 충돌 방지
    if "eva_init" not in ss:
        ss.eva_init = True
        ss.eva_recording = False
        ss.eva_stopped = False
        ss.eva_auto_saved_once = False
        ss.eva_last_wav = None
        ss.eva_qidx = 0
        ss.eva_questions = list(questions)
        ss.eva_last_stt = ""
        ss.eva_last_fb = ""
        ss.setdefault("eva_pending_analysis", False)  # ★ 다음 런에서 분석 실행 여부
        ss.setdefault("eva_history", [])
        ss.setdefault("eva_pending_final", False)
        
    def interviewer_line(q: str) -> str:
        return f"질문 드리겠습니다. {q} 답변을 시작하신 뒤, 완료되면 종료 버튼을 눌러 주세요."

    c1, c2, c3 = st.columns([1, 1, 1])

    # 1) 면접 시작/다음 질문 (TTS 재생)
    with c1:
        if st.button("🎤 면접 시작 / 다음 질문", use_container_width=True, disabled=ss.eva_recording):
            q = ss.eva_questions[ss.eva_qidx]
            try:
                audio_bytes, mime = tts_interviewer(interviewer_line(q), tts_speed)
                st.audio(audio_bytes, format=mime)  # "audio/wav" or "audio/mp3"
                st.toast(f"Q{ss.eva_qidx + 1} 재생 완료")
            except Exception as e:
                st.error(f"TTS 재생 실패: {e}")

    with c2:
        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("▶️ 답변 시작", use_container_width=True, disabled=ss.eva_recording, key="btn_start"):
        
                    COUNTDOWN = 5  # 지연(초)
                    msg = st.empty()
                    bar = st.progress(0, text="곧 면접을 시작합니다…")
                    try:
                        for sec in range(COUNTDOWN, 0, -1):
                            msg.info(f"🕒 {sec}초 후 답변 시작")
                            bar.progress(int(((COUNTDOWN - sec + 1) / COUNTDOWN) * 100))
                            time.sleep(1)

                        # ⬇️ 카운트다운 끝난 뒤 실제 녹음 시작
                        r = requests.get(f"{server_url}/command/start_record", timeout=(3.05, 8))
                        r.raise_for_status()

                        ss.eva_recording = True
                        ss.eva_stopped = False
                        ss.eva_auto_saved_once = False
                        ss.eva_last_wav = None

                        st.success("면접이 시작했습니다. 답변을 말씀해 주세요.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"요청 실패: {e}")
                    finally:
                        bar.empty()
                        msg.empty()

        with col_stop:
            if st.button("⏹️ 답변 종료", use_container_width=True, disabled=not ss.eva_recording, key="btn_stop"):
                try:
                    # 1) 녹음 종료
                    r = requests.get(f"{server_url}/command/stop_record", timeout=60)
                    r.raise_for_status()

                    # ★ 상태 업데이트만 하고, 긴 작업은 다음 런에서 수행
                    ss.eva_recording = False
                    ss.eva_stopped = True
                    ss.eva_pending_analysis = True   # 다음 런에서 run_analysis_if_needed()가 실행
                    ss.eva_auto_saved_once = False
                    st.success("답변이 종료되었습니다. 분석을 시작합니다…")

                    # ★ 즉시 UI 재렌더 → '답변 시작' 버튼이 바로 활성화됨
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"요청 실패: {e}")
    # 3) 면접 종료(세션 종료)
    with c3:
        end_enabled = (not ss.eva_recording) and (len(ss.eva_history) > 0 or ss.eva_pending_analysis)
        if st.button("🏁 면접 종료", use_container_width=True, disabled=not end_enabled, key="btn_end"):
            ss.eva_pending_final = True    # ✅ 총평을 실행하라는 신호만 남기기
            st.rerun()                     # ✅ 다음 런에서 총평 블록 실행
    # 면접 내용 저장(mp4, wav, xml)
    if ss.eva_stopped and not ss.eva_auto_saved_once:
        kinds = ["wav", "xml"] if ss.get("mp4_manual_only") else ["mp4", "wav", "xml"]
        with st.spinner("저장 중…"):
            saved = save_assets_after_stop(server_url, ss.get("session_id", "sess"), kinds=kinds)
        if saved:
            ss.eva_auto_saved_once = True
            st.success("저장 완료!")

    answer_box = st.container()  # 방금 답변 결과(텍스트/피드백만)
    final_box  = st.container()  # 면접 종료 총평 표시
    if ss.eva_pending_analysis:
             with answer_box:
                try:
                    # 1) 경로 확보 (없으면 다운로드)
                    if not ss.eva_last_wav:
                        wav_path = download_wav_direct(server_url, max_wait_s=30, interval_s=0.5)
                        ss.eva_last_wav = str(wav_path)

                    # 2) (보너스) 파일이 없어졌으면 다시 다운로드
                    import os
                    if not os.path.exists(ss.eva_last_wav):
                        wav_path = download_wav_direct(server_url, max_wait_s=30, interval_s=0.5)
                        ss.eva_last_wav = str(wav_path)

                    # 3) 유효성 검사 → STT(경로 기반)
                    st.info("WAV 유효성 검사…")
                    ok, info = is_valid_wav(ss.eva_last_wav)
                    if not ok:
                        st.error(f"오디오 파일 이상: {info}")
                        ss.eva_pending_analysis = False
                        st.stop()

                    st.info("STT 진행 중…")
                    ss.eva_last_stt = stt_fn(ss.eva_last_wav)   # my_stt_from_path 사용

                    qtext = ss.eva_questions[ss.eva_qidx] if ss.eva_questions else f"Q{ss.eva_qidx+1}"
                    st.info("피드백 생성 중…")
                    ss.eva_last_fb = feedback_fn(qtext, ss.eva_last_stt)

                    # (선택) 음정/자세 계산해서 저장만
                    pitch_metrics, posture_metrics = {}, {}
                    try:
                        feats, _ = analyze_stability(ss.eva_last_wav)
                        score, label, _ = get_stability_score(
                            feats.get("jitter", 0.0),
                            feats.get("shimmer", 0.0),
                            feats.get("hnr", 0.0),
                        )
                        pitch_metrics = {**feats, "stability_score": score, "stability_label": label}
                    except Exception:
                        pass
                    try:
                        xml_path = resolve_posture_xml_for(ss.eva_last_wav)
                        if xml_path:
                            posture_metrics = parse_posture_summary(xml_path)
                    except Exception:
                        pass

                    # 화면 출력
                    st.markdown("### 📝 이번 답변")
                    st.write(ss.eva_last_stt or "(빈 텍스트)")
                    st.markdown("### 🎯 피드백")
                    st.write(ss.eva_last_fb or "(피드백 없음)")

                    # ✅ 히스토리 저장
                    ss.eva_history.append({
                        "qidx": ss.eva_qidx,
                        "qtext": qtext,
                        "wav_path": ss.eva_last_wav,
                        "stt": ss.eva_last_stt,
                        "fb": ss.eva_last_fb,
                        "pitch": pitch_metrics,
                        "posture": posture_metrics,
                    })
                    if ss.eva_questions and ss.eva_qidx < len(ss.eva_questions) - 1:
                        ss.eva_qidx += 1

                    st.success("분석 완료")
                except Exception as e:
                    st.exception(e)
                finally:
                    ss.eva_pending_analysis = False  # ✅ 여기서 확실히 내리기

                 # ✅ 방금 분석을 끝냈고 사용자가 미리 🏁 눌러둔 경우에만 1번 rerun
    if ss.eva_pending_final and not ss.get("_reran_to_final_once", False):
        ss["_reran_to_final_once"] = True
        st.rerun()

    elif ss.eva_pending_final and not ss.eva_pending_analysis:
        with final_box:
            try:
                import statistics as stats, json
                # --- (그대로 유지) 지표 집계 ---
                def _avg(seq):
                    xs = [x for x in seq if x is not None]
                    return stats.mean(xs) if xs else None

                getp = lambda k: [h.get("pitch",{}).get(k) for h in ss.eva_history]
                summary_voice = {
                    "avg_stability_score": _to_native(_avg(getp("stability_score"))),
                    "avg_jitter":          _to_native(_avg(getp("jitter"))),
                    "avg_shimmer":         _to_native(_avg(getp("shimmer"))),
                    "avg_hnr":             _to_native(_avg(getp("hnr"))),
                    "avg_f0_std":          _to_native(_avg(getp("f0_std"))),
                    "avg_loudness_std":    _to_native(_avg(getp("loudness_std"))),
                }

                total_frames = sum((h.get("posture",{}).get("frames") or 0) for h in ss.eva_history)
                sum_head    = sum((h.get("posture",{}).get("head_tilt_count") or 0) for h in ss.eva_history)
                sum_body    = sum((h.get("posture",{}).get("body_tilt_count") or 0) for h in ss.eva_history)
                sum_gesture = sum((h.get("posture",{}).get("gesture_count") or 0) for h in ss.eva_history)
                def _rate(n, d): return (n / d) if d and n is not None else None
                summary_posture = {
                    "head_tilt_rate":    _to_native(_rate(sum_head, total_frames)),
                    "body_tilt_rate":    _to_native(_rate(sum_body, total_frames)),
                    "gesture_per_frame": _to_native(_rate(sum_gesture, total_frames)),
                    "frames_total":      int(total_frames),
                }

                # --- (화면 출력 X) 지표는 LLM 프롬프트로만 전달 ---
                history_compact = [{
                    "q": h["qtext"],
                    "stt": (h.get("stt","")[:150] + "…") if h.get("stt") and len(h["stt"])>150 else (h.get("stt") or "")
                } for h in ss.eva_history[-8:]]

                messages = get_prompt("session").format_messages(
                    voice_summary_json=json.dumps(summary_voice, ensure_ascii=False),
                    posture_summary_json=json.dumps(summary_posture, ensure_ascii=False),
                    history_compact_json=json.dumps(history_compact, ensure_ascii=False),
                    answers=len(ss.eva_history),
                )

                summary_text = call_llm(messages)
                st.markdown("## 🧾 면접 총평")
                st.write(summary_text)

                st.success("면접 총평 생성 완료")

            except Exception as e:
                # LLM 실패 시에도 지표 박스는 안 보여주고, 간단 안내만
                st.markdown("## 🧾 면접 총평")
                st.warning("총평 생성에 문제가 발생했어요. 잠시 후 다시 시도해 주세요.")
                st.info(f"(참고: {e})")

            finally:
                ss.eva_pending_final = False
                ss["_reran_to_final_once"] = False  # (있다면) rerun 가드 해제

    # 상태 표시
    st.markdown("🟢 **답변 중입니다...**" if ss.eva_recording else "⚪ **면접 대기 중**")
    st.markdown("---")
    st.subheader(f"Q{ss.eva_qidx + 1}. {ss.eva_questions[ss.eva_qidx]}")

    # 결과 표시
    if ss.eva_last_stt:
        st.markdown("**STT 결과**"); st.write(ss.eva_last_stt)
    if ss.eva_last_fb:
        st.markdown("**피드백**"); st.write(ss.eva_last_fb)

    # 다음 질문 이동
    if st.button("➡️ 다음 질문으로", disabled=ss.eva_recording):
        ss.eva_qidx = (ss.eva_qidx + 1) % len(ss.eva_questions)
        ss.eva_last_wav = None
        ss.eva_stopped = False
        ss.eva_auto_saved_once = False
        st.toast(f"Q{ss.eva_qidx + 1}로 이동")
