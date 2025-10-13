import os
from flask import Flask, request, send_from_directory

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

if __name__ == '__main__':
    # host='0.0.0.0'는 모든 IP에서 접속 가능하게 합니다.
    app.run(host='10.10.14.80', port=5000)