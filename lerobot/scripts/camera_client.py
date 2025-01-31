import json
import threading
from io import StringIO
import PIL
from PIL import Image, ImageTk

import cv2
import numpy as np
import socket
import pickle
import time
import tkinter as tk

class CameraClient:
    def run(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.connect(("50.39.109.27", 50064))
        self.buffer_img = None
        self.current_img = None

        server_thread = threading.Thread(target=self.client_thread, args=[self.server])
        server_thread.start()

        self.root = tk.Tk()
        self.label = tk.Label(self.root, text="Waiting for first image")
        self.label.pack()

        self.root.after(10, self.update_window)
        self.root.mainloop()

    def client_thread(self, server):
        #
        data = {}
        data['control_mode'] = "remote_stream"
        data['camera_name'] = "laptop"
        json_data = json.dumps(data)
        server.sendall(json_data.encode())

        buffer = b''
        server.setblocking(False)

        while True:
            try:
                recv_data = server.recv(4096)
                buffer += recv_data
            except BlockingIOError:
                if buffer == b'':
                    continue
                
                pieces = buffer.split(b'json_over')
                buffer = b''

                print(pieces)

                # print("got a frame")

                # data = pieces[0]
                # frame = np.asarray(bytearray(data))
                # self.buffer_img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                # print(self.buffer_img)


    def update_window(self):
        while self.buffer_img is None:
            continue

        im = PIL.Image.fromarray(self.buffer_img)
        self.current_img = PIL.ImageTk.PhotoImage(image = im)

        self.label.config(image=self.current_img)
        self.label.pack()
        self.root.after(10, self.update_window)

if __name__ == "__main__":
    client = CameraClient()
    client.run()
