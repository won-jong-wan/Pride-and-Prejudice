import subprocess
import time
import sys

def process_start():
    global estimator, recorder
    
    # 영상 데이터 저장을 처리하는 파이썬 스크립트 실행
    recorder = subprocess.Popen(
        [sys.executable, "recording.py"], # 현재 파이썬 실행기로 스크립트 실행
        stdin=subprocess.PIPE,      # 파이썬에서 입력을 보낼 통로
        stdout=subprocess.DEVNULL,     # 프로세스의 출력을 읽어올 통로
        stderr=subprocess.PIPE,
        text=True,  # 입출력을 텍스트(str)로 처리
        bufsize=1   # 출력을 버퍼링 없이 바로바로 받기 위함
    )

    # run.sh 스크립트를 실행하는 프로세스
    estimator = subprocess.Popen(["bash", "run.sh"], stdout=subprocess.DEVNULL) # 출력은 보지 않음

# r: 시작
# q: 종료
def recorder_send_cmd(cmd):
    global recorder
    if recorder.poll() is None: # 프로세스가 아직 종료되지 않았으면
        recorder.stdin.write(cmd + "\n") # 명령어 전송
        recorder.stdin.flush()            # 버퍼 비우기

def estimator_finish():
    global estimator
    if estimator.poll() is None: # 프로세스가 아직 종료되지 않았으면
        estimator.kill()    # 프로세스 종료
        # print("Estimator process terminated.")
    
def recorder_finish():
    global recorder
    if recorder.poll() is None: # 프로세스가 아직 종료되지 않았으면
        recorder.terminate()    # 프로세스 종료

if __name__ == '__main__':
    process_start()
    # time.sleep(5)  # 5초 대기
    # recorder_send_cmd('r')  # 녹음 시작 명령어 전송
    # time.sleep(10) # 10초 녹음
    # recorder_send_cmd('q')  # 녹음 종료 명령어 전송
    # estimator_finish()      # estimator 프로세스 종료
    # recorder_finish()       # recorder 프로세스 종료
    # print("Processes finished.")