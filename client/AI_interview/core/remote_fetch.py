import time, requests
import streamlit as st

def wait_until_ready(url: str, max_wait_s: int = 8, interval_s: float = 0.5):
    """
    서버 파일 준비 확인:
      - HEAD 시도
      - 실패 시 Range GET(0-0)로 존재 확인
      - 준비되면 최종 GET으로 전체 바이트 반환
    """
    deadline = time.time() + max_wait_s
    last_err = None

    while time.time() < deadline:
        try:
            h = requests.head(url, timeout=2)
            if h.status_code == 200 and int(h.headers.get("Content-Length", "0")) > 0:
                g = requests.get(url, timeout=10, allow_redirects=True)
                if g.ok and g.content:
                    return g

            r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=3, allow_redirects=True, stream=True)
            if r.status_code in (200, 206):
                g = requests.get(url, timeout=10, allow_redirects=True)
                if g.ok and g.content:
                    return g
        except Exception as e:
            last_err = e
        time.sleep(interval_s)

    if last_err:
        st.info(f"파일 확인 중 오류: {last_err}")
    return None
