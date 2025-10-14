import os
from flask import Flask, request, send_from_directory
from scripts.process_call import process_start, rtsp_server_finish, estimator_finish, recorder_finish

# 'uploads' 라는 폴더를 만들고 파일을 여기 저장할 겁니다.
FOLDER = '../srv_tmp'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

app = Flask(__name__)
app.config['FOLDER'] = FOLDER

# 파일 업로드 처리
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'no file', 400
    file = request.files['file']
    if file.filename == '':
        return 'no file selected', 400
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['FOLDER'], filename))
        return f"'{filename}' file upload successful!", 200

# File download processing (http://<Raspberry_Pi_IP>:5000/download/subfolder/filename)
@app.route('/download/<path:filepath>')
def download_file(filepath):
    # 파일 경로에서 디렉토리와 파일 이름 분리
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # 기본 폴더와 하위 경로 결합
    full_directory = os.path.join(app.config['FOLDER'], directory) if directory else app.config['FOLDER']
    
    return send_from_directory(full_directory, filename, as_attachment=True)

# http://<Raspberry_Pi_IP>:5000/command/command 형식으로 GET 요청을 받습니다.
@app.route('/command/<cmd>')
def execute_command(cmd):
    print(f"'{cmd}' command received.")
    
    # 'start_record' command received
    if cmd == 'start_record':
        # 여기에 실제 녹음 시작 코드를 넣으면 됩니다.
        print(">> start recording!")
        process_start()  # 프로세스 시작
        return f"'{cmd}' command executed successfully!", 200

    # 'stop_record' command received
    elif cmd == 'stop_record':
        print(">> stop recording!")
        recorder_finish()       # recorder 프로세스 종료
        estimator_finish()      # estimator 프로세스 종료
        rtsp_server_finish()    # rtsp_server 프로세스 종료
        return f"'{cmd}' command executed successfully!", 200

    else:
        return "Unknown command.", 400

if __name__ == '__main__':
    # host='0.0.0.0' allows access from all IPs.
    app.run(host='10.10.14.80', port=5000)