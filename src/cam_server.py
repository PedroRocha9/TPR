from flask import Flask, request, Response, render_template_string
import threading
import time
from cryptography.fernet import Fernet

app = Flask(__name__)
frame_lock = threading.Lock()
current_frame = None

# Load the key
with open('secret.key', 'rb') as key_file:
    key = key_file.read()
fernet = Fernet(key)

@app.route('/upload', methods=['POST'])
def upload():
    global current_frame
    file = request.files['file']
    encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    with frame_lock:
        current_frame = decrypted_data
    return 'Frame received'

def gen_frames():
    while True:
        with frame_lock:
            if current_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.1)  # Adjust the sleep time as needed

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
    <title>Webcam Stream</title>
    </head>
    <body>
    <h1>Webcam Stream</h1>
    <img src="/video_feed">
    </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)

