import json
from io import StringIO

import cv2
import numpy as np
import socket
import pickle
import time

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.connect(("50.39.109.27", 50064))
#
data = {}
data['control_mode'] = "remote_stream"
data['camera_name'] = "phone"
json_data = json.dumps(data)
server.sendall(json_data.encode())

buffer = b''
start = time.perf_counter()
while True:
    recv_data = server.recv(4096)
    buffer += recv_data


    if buffer.endswith(b'this_is_the_end'):
        pieces = buffer.split(b'this_is_the_end')
        buffer = b''

        data = pieces[0]
        frame = np.asarray(bytearray(data))
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Camera", frame)
        # print("Image Displayed, Spent " + str(time.perf_counter() - start) + "s recieving")
        start = time.perf_counter()
        response = "img_recieved"
        server.send(response.encode())

    if cv2.waitKey(1) == ord('q'):
        break