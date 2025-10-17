import subprocess
import time
import sys
import signal
import os

# def process_start():
#     global rtsp_server, p_estimator, recorder

#     time.sleep(1)

#     rtsp_server = subprocess.Popen(
#         [sys.executable, '-m', 'scripts.gstreamer.rtsp', '8554', '90'],
#         stdout=subprocess.DEVNULL
#     )

#     time.sleep(1)  # RTSP 서버가 시작될 시간을 줍니다.
    
#     # 영상 데이터 저장을 처리하는 파이썬 스크립트 실행
#     recorder = subprocess.Popen(
#         [sys.executable, '-m', 'scripts.gstreamer.recorder'], # 현재 파이썬 실행기로 스크립트 실행
#         stdout=subprocess.DEVNULL    # 프로세스의 출력을 읽어올 통로
#     )

#     p_estimator = subprocess.Popen(
#         [sys.executable, '-m', 'scripts.pose_est.pose_est_main',
#         '--hef', 'models/vit_pose_small.hef',
#         '--camera', 'rtsp://127.0.0.1:8554/test',
#         '--conf', '0.4',
#         '--width', '192',
#         '--height', '256'], 
#         stdout=subprocess.DEVNULL, 
#         preexec_fn=os.setsid
#     )

class ProcessManager:
    def __init__(self):
        self.rtsp_server = None
        self.f_estimator = None
        self.p_estimator = None
        self.recorder = None

    def rtsp_start(self):
        self.rtsp_server = subprocess.Popen(
            [sys.executable, '-m', 'scripts.gstreamer.rtsp', '8554', '90'],
            stdout=subprocess.DEVNULL
        )

    def f_estimator_start(self):
        self.f_estimator = subprocess.Popen(
            [sys.executable, '-m', 'scripts.face_est.face_est_main',
            '--camera', 'rtsp://127.0.0.1:8554/test'], 
            stdout=subprocess.DEVNULL, 
        )

    def p_estimator_start(self):
        self.p_estimator = subprocess.Popen(
            [sys.executable, '-m', 'scripts.pose_est.pose_est_main',
            '--hef', 'models/vit_pose_small.hef',
            '--camera', 'rtsp://127.0.0.1:8554/test',
            '--conf', '0.4',
            '--width', '192',
            '--height', '256'], 
            stdout=subprocess.DEVNULL, 
            preexec_fn=os.setsid
        )

    def recorder_start(self):
        self.recorder = subprocess.Popen(
            [sys.executable, '-m', 'scripts.gstreamer.recorder'], 
            stdout=subprocess.DEVNULL
        )
        print("Recorder started.")

    def rtsp_server_finish(self):
        self.rtsp_server.send_signal(signal.SIGINT)

    def f_estimator_finish(self):
        self.f_estimator.send_signal(signal.SIGINT)

    def p_estimator_finish(self):
        self.p_estimator.send_signal(signal.SIGINT)

        os.killpg(os.getpgid(self.p_estimator.pid), signal.SIGTERM)

    def recorder_finish(self):
        self.recorder.send_signal(signal.SIGINT)

# if __name__ == '__main__':
#     process_start()
#     # time.sleep(5)  # 5초 대기
#     # recorder_send_cmd('r')  # 녹음 시작 명령어 전송
#     # time.sleep(10) # 10초 녹음
#     # recorder_send_cmd('q')  # 녹음 종료 명령어 전송
#     # estimator_finish()      # estimator 프로세스 종료
#     # recorder_finish()       # recorder 프로세스 종료
#     # print("Processes finished.")
