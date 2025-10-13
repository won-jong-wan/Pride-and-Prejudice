import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# GStreamer 및 RTSP 서버 초기화
Gst.init(None)

class TestServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)

    # 클라이언트가 접속할 때마다 호출되어 파이프라인을 생성하는 함수
    def do_create_element(self, url):
        # 웹캠 영상을 H.264로 인코딩하여 RTP로 보내는 파이프라인
        # gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! x264enc tune=zerolatency ! rtph264pay name=pay0 pt=96
        pipeline_str = (
            "v4l2src device=/dev/video0 "
            "! videoconvert "
            "! x264enc tune=zerolatency "
            "! rtph264pay name=pay0 pt=96"
        )
        print(f"Server pipeline: {pipeline_str}")
        return Gst.parse_launch(pipeline_str)

class GstServer():
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        
        # 기본 포트는 8554. 변경하려면 self.server.set_service("포트번호") 사용
        # self.server.set_service("8554") 

        # 서버에 URL 마운트 포인트를 추가
        mounts = self.server.get_mount_points()
        
        # /test 라는 엔드포인트에 위에서 정의한 팩토리를 연결
        factory = TestServerFactory()
        factory.set_shared(True) # 여러 클라이언트가 동일한 스트림을 공유하도록 설정
        mounts.add_factory("/test", factory)
        
        self.server.attach(None)
        
        print("RTSP stream ready at rtsp://127.0.0.1:8554/test")
        print("Press Ctrl+C to stop the server.")

if __name__ == '__main__':
    s = GstServer()
    loop = GLib.MainLoop()
    loop.run()