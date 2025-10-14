from functools import partial
import time
import numpy as np
from ..pose_est.pose_analyzer_kalman import KalmanPoseAnalyzer
from ..pose_est.pose_logger import create_pose_logger

class ParallelInferenceManager:
    def __init__(self, hailo_manager, camera_manager, conf_threshold=0.3):
        """
        병렬 추론 관리 클래스
        
        Args:
            hailo_manager: HailoDeviceManager 인스턴스
            camera_manager: CameraManager 인스턴스
            conf_threshold: Confidence threshold
        """
        self.hailo_manager = hailo_manager
        self.camera_manager = camera_manager
        self.conf_threshold = conf_threshold
        
        # 결과 저장용
        self.results = {'model1': None, 'model2': None}
        self.frames_ready = 0
        
        # Pose Analyzer 초기화
        self.pose_analyzer = KalmanPoseAnalyzer(
            window_size=45,
            process_noise=0.001,
            measurement_noise=0.7
        )
        
        # XML 로거 초기화
        self.pose_logger = create_pose_logger(fps=30)
    
    def inference_callback(self, completion_info, model_name, bindings):
        """비동기 추론 완료 콜백"""
        if completion_info.exception:
            print(f"Error during {model_name} inference: {completion_info.exception}")
            return
        
        try:
            # 결과 저장 - get_buffer() 메서드를 사용하여 출력 데이터 접근
            self.results[model_name] = bindings.output().get_buffer()
            self.frames_ready += 1
            
        except Exception as e:
            print(f"Error in {model_name} callback: {e}")
    
    def run_inference(self, frame):
        """단일 프레임에 대한 추론 실행"""
        # 프레임 전처리
        input_data = self.camera_manager.preprocess_frame(frame)
        
        # 바인딩 생성
        bindings1, bindings2 = self.hailo_manager.create_bindings()
        
        # 입력 데이터 설정
        bindings1.input("vit_pose_small/input_layer1").set_buffer(input_data)
        bindings2.input("mspn_regnetx_800mf/input_layer1").set_buffer(input_data)
        
        # 결과를 저장할 버퍼 설정
        output_shape1 = self.hailo_manager.model1.output().shape
        output_shape2 = self.hailo_manager.model2.output().shape
        bindings1.output().set_buffer(np.empty(output_shape1).astype(np.uint8))
        bindings2.output().set_buffer(np.empty(output_shape2).astype(np.uint8))
        
        # 파이프라인 준비 대기
        self.hailo_manager.wait_for_ready()
        
        # 비동기 추론 시작
        start_time = time.time()
        
        job1 = self.hailo_manager.configured_model1.run_async(
            [bindings1],
            partial(self.inference_callback, model_name='model1', bindings=bindings1)
        )
        
        job2 = self.hailo_manager.configured_model2.run_async(
            [bindings2],
            partial(self.inference_callback, model_name='model2', bindings=bindings2)
        )
        
        # 두 작업이 완료될 때까지 대기
        try:
            job1.wait(timeout_ms=self.hailo_manager.timeout_ms)
            job2.wait(timeout_ms=self.hailo_manager.timeout_ms)
        except Exception as e:
            print(f"Error waiting for inference jobs: {e}")
            return 0  # inference_time만 반환하므로 0을 반환
        
        inference_time = (time.time() - start_time) * 1000
        return inference_time
    
    def process_results(self, frame):
        """추론 결과 처리 및 시각화"""
        if self.results['model1'] is not None:
            # 첫 번째 모델 결과 처리
            keypoints1 = self.process_model1_output(self.results['model1'], frame.shape[:2])
            pose_analysis = self.pose_analyzer.analyze_pose(keypoints1)
            
            # 로그 저장
            self.pose_logger.log_analysis(pose_analysis)
            
            return pose_analysis
        return None
    
    def process_model1_output(self, output, image_size):
        """첫 번째 모델(포즈 추정) 출력 처리"""
        # 여기에 구체적인 후처리 로직 구현
        # 현재는 간단한 예시만 포함
        return output
    
    def cleanup(self):
        """리소스 정리"""
        self.pose_logger.close()
        self.hailo_manager.cleanup()
        self.camera_manager.cleanup()