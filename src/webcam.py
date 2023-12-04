import cv2
import requests
import time
from cryptography.fernet import Fernet

# Load the key
with open('secret.key', 'rb') as key_file:
    key = key_file.read()
fernet = Fernet(key)

SERVER_URL = 'http://localhost:8081/upload'

def send_frame_to_server(frame):
    _, encoded_image = cv2.imencode('.jpg', frame)
    encrypted_image = fernet.encrypt(encoded_image.tobytes())
    files = {'file': ('image.jpg', encrypted_image, 'image/jpeg', {'Expires': '0'})}
    try:
        response = requests.post(SERVER_URL, files=files)
        print(response.text)
    except Exception as e:
        print(f"Error sending frame: {e}")

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        send_frame_to_server(frame)
        time.sleep(0.1)

    cap.release()

if __name__ == "__main__":
    main()
