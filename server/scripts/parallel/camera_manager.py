import cv2
import numpy as np
from collections import deque
import time

class CameraManager:
    def __init__(self, camera_device='/dev/video0', input_size=(192, 256)):
        """
        카메라 관리 및 전처리를 위한 클래스
        
        Args:
            camera_device: 카메라 디바이스 경로
            input_size: 모델 입력 크기 (width, height)
        """
        self.input_size = input_size
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        self._initialize_camera(camera_device)
    
    def _initialize_camera(self, camera_device):
        """카메라 초기화 및 설정"""
        print(f"Opening camera: {camera_device}")
        self.cap = cv2.VideoCapture(camera_device)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_device}")
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
    
    def read_frame(self):
        """프레임 읽기"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def preprocess_frame(self, frame):
        """프레임 전처리"""
        img = cv2.resize(frame, self.input_size)
        img = np.expand_dims(img, axis=0)  # batch dimension 추가
        return img.astype(np.uint8)
    
    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)
    
    def draw_info(self, frame, fps, inference_time, pose_analysis):
        """정보 오버레이 표시"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # 기본 정보
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 분석 결과 표시
        y_pos = 100
        for key, (is_detected, msg) in pose_analysis.items():
            color = (0, 0, 255) if is_detected else (0, 255, 0)
            cv2.putText(frame, msg, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
        
        return frame
    
    def cleanup(self):
        """리소스 정리"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()