import argparse
import cv2
from .hailo_manager import HailoDeviceManager
from .camera_manager import CameraManager
from .inference_manager import ParallelInferenceManager

def main():
    parser = argparse.ArgumentParser(description='Parallel Model Inference with Hailo')
    parser.add_argument('--hef1', type=str, required=True,
                       help='Path to first HEF model file')
    parser.add_argument('--hef2', type=str, required=True,
                       help='Path to second HEF model file')
    parser.add_argument('--camera', type=str, default='/dev/video0',
                       help='Camera device path (default: /dev/video0)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--width', type=int, default=192,
                       help='Model input width (default: 192)')
    parser.add_argument('--height', type=int, default=256,
                       help='Model input height (default: 256)')
    
    args = parser.parse_args()
    
    try:
        # 각 매니저 초기화
        hailo_manager = HailoDeviceManager(
            hef_path1=args.hef1,
            hef_path2=args.hef2
        )
        
        camera_manager = CameraManager(
            camera_device=args.camera,
            input_size=(args.width, args.height)
        )
        
        inference_manager = ParallelInferenceManager(
            hailo_manager=hailo_manager,
            camera_manager=camera_manager,
            conf_threshold=args.conf
        )
        
        print("\nStarting parallel inference...")
        print("Press 'q' to quit\n")
        
        while True:
            # 프레임 읽기
            frame = camera_manager.read_frame()
            if frame is None:
                break
            
            # 추론 실행
            inference_time = inference_manager.run_inference(frame)
            
            # 결과 처리
            pose_analysis = inference_manager.process_results(frame)
            
            if pose_analysis:
                # FPS 계산
                fps = camera_manager.calculate_fps()
                
                # 결과 시각화
                frame = camera_manager.draw_info(
                    frame, fps, inference_time, pose_analysis
                )
                
                # 결과 표시
                cv2.imshow('Parallel Inference', frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        print("\nCleaning up...")
        inference_manager.cleanup()
        print("Done!")

if __name__ == '__main__':
    main()