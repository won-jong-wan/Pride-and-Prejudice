import subprocess
import time
import sys
import signal
import os

def process_start():
    global rtsp_server, estimator, recorder

    time.sleep(1)

    rtsp_server = subprocess.Popen(
        [sys.executable, "scripts/gstreamer/rtsp.py"],
        stdout=subprocess.DEVNULL
    )

    time.sleep(2)
    
    # 영상 데이터 저장을 처리하는 파이썬 스크립트 실행
    recorder = subprocess.Popen(
        [sys.executable, "scripts/gstreamer/recorder.py"], # 현재 파이썬 실행기로 스크립트 실행
        stdout=subprocess.DEVNULL    # 프로세스의 출력을 읽어올 통로
    )

    # run.sh 스크립트를 실행하는 프로세스
    estimator = subprocess.Popen(
        ["bash", "run.sh"], 
        stdout=subprocess.DEVNULL, 
        preexec_fn=os.setsid
    ) # 출력은 보지 않음

def rtsp_server_finish():
    global rtsp_server

    rtsp_server.send_signal(signal.SIGINT)

def estimator_finish():
    global estimator

    os.killpg(os.getpgid(estimator.pid), signal.SIGTERM)
    
def recorder_finish():
    global recorder

    recorder.send_signal(signal.SIGINT)

if __name__ == '__main__':
    process_start()
    # time.sleep(5)  # 5초 대기
    # recorder_send_cmd('r')  # 녹음 시작 명령어 전송
    # time.sleep(10) # 10초 녹음
    # recorder_send_cmd('q')  # 녹음 종료 명령어 전송
    # estimator_finish()      # estimator 프로세스 종료
    # recorder_finish()       # recorder 프로세스 종료
    # print("Processes finished.")