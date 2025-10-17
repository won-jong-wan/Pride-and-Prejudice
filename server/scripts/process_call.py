import subprocess
import time
import sys
import signal
import os

class ProcessManager:
    def __init__(self):
        self.rtsp_server = None
        self.f_estimator = None
        self.p_estimator = None
        self.recorder = None

    def rtsp_start(self, target):
        if target == 'file':
            rtsp_name = 'rtsp_file'
        elif target == 'a6700':
            rtsp_name = 'rtsp_a6700'
        elif target == 'webcam':
            rtsp_name = 'rtsp_webcam'

        self.rtsp_server = subprocess.Popen(
            [sys.executable, '-m', f'scripts.gstreamer.{rtsp_name}'],
            stdout=subprocess.PIPE
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

    def xml_mix(self):
        self.xml_mixer = subprocess.Popen(
            [sys.executable, '-m', 'scripts.xml_mix'],
            stdout=subprocess.DEVNULL
        )

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
