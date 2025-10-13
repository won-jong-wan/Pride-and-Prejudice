import os
from flask import Flask, request, send_from_directory
from process_call import process_start, recorder_send_cmd, estimator_finish, recorder_finish

# 'uploads' 라는 폴더를 만들고 파일을 여기 저장할 겁니다.
FOLDER = 'tmp'
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

# File download processing (http://<Raspberry_Pi_IP>:5000/download/filename)
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['FOLDER'], filename)

# http://<Raspberry_Pi_IP>:5000/command/command 형식으로 GET 요청을 받습니다.
@app.route('/command/<cmd>')
def execute_command(cmd):
    print(f"'{cmd}' command received.")
    
    # 'start_record' command received
    if cmd == 'start_record':
        # 여기에 실제 녹음 시작 코드를 넣으면 됩니다.
        print(">> start recording!")
        process_start()  # 프로세스 시작
        recorder_send_cmd('r')  # 녹음 시작 명령어 전송
        return f"'{cmd}' command executed successfully!", 200

    # 'stop_record' command received
    elif cmd == 'stop_record':
        print(">> stop recording!")
        recorder_send_cmd('q')  # 녹음 종료 명령어 전송
        estimator_finish()      # estimator 프로세스 종료
        return f"'{cmd}' command executed successfully!", 200

    else:
        return "Unknown command.", 400

if __name__ == '__main__':
    # host='0.0.0.0' allows access from all IPs.
    app.run(host='10.10.14.80', port=5000)