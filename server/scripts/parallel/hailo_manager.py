from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType

class HailoDeviceManager:
    def __init__(self, hef_path1, hef_path2, timeout_ms=10000):
        """
        Hailo 디바이스 및 모델 관리 클래스
        
        Args:
            hef_path1: 첫 번째 HEF 모델 파일 경로
            hef_path2: 두 번째 HEF 모델 파일 경로
            timeout_ms: 추론 타임아웃 (밀리초)
        """
        self.timeout_ms = timeout_ms
        self.vdevice = None
        self.model1 = None
        self.model2 = None
        self.configured_model1 = None
        self.configured_model2 = None
        
        self._initialize_device()
        self._load_models(hef_path1, hef_path2)
    
    def _initialize_device(self):
        """Hailo 디바이스 초기화"""
        print("Initializing Hailo device...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
    
    def _load_models(self, hef_path1, hef_path2):
        """HEF 모델 로드 및 설정"""
        print(f"Loading models: {hef_path1}, {hef_path2}")
        
        # 모델 로드
        self.model1 = self.vdevice.create_infer_model(hef_path1)
        self.model2 = self.vdevice.create_infer_model(hef_path2)
        
        # 입력/출력 포맷 설정 (UINT8 사용)
        self.model1.input("vit_pose_small/input_layer1").set_format_type(FormatType.UINT8)
        self.model1.output().set_format_type(FormatType.UINT8)
        
        self.model2.input("mspn_regnetx_800mf/input_layer1").set_format_type(FormatType.UINT8)
        self.model2.output().set_format_type(FormatType.UINT8)
        
        # 모델 configure
        self.configured_model1 = self.model1.configure()
        self.configured_model2 = self.model2.configure()
    
    def create_bindings(self):
        """두 모델에 대한 바인딩 생성"""
        bindings1 = self.configured_model1.create_bindings()
        bindings2 = self.configured_model2.create_bindings()
        return bindings1, bindings2
    
    def wait_for_ready(self):
        """비동기 파이프라인 준비 대기"""
        self.configured_model1.wait_for_async_ready(timeout_ms=self.timeout_ms)
        self.configured_model2.wait_for_async_ready(timeout_ms=self.timeout_ms)
    
    def cleanup(self):
        """리소스 정리"""
        # 필요한 경우 Hailo 리소스 정리
        pass