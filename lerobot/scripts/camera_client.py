import json
from io import StringIO

import cv2
import numpy as np
import socket
import pickle
import time
import sys

mouseX, mouseY = (0,0)

def update_coords(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y

if __name__ == "__main__":

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect(("50.39.109.27", 50064))
    #
    data = {}
    data['control_mode'] = "remote_stream"
    data['camera_name'] = sys.argv[1]
    json_data = json.dumps(data)
    server.sendall(json_data.encode())

    buffer = b''
    start = time.perf_counter()

    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', update_coords)

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
            response = (mouseX, mouseY)
            server.send(pickle.dumps(response))

        if cv2.waitKey(1) == ord('q'):
            break