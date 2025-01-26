import socket
import pickle
import threading
from io import StringIO

import cv2
import numpy as np
from time import sleep

def accept_client(client: socket):
    camera = cv2.VideoCapture(8)
    while True and not time_to_stop:
        ret, frame = camera.read()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        result, encimg = cv2.imencode('.jpg', frame, encode_param)

        client.sendall(np.array(encimg).tobytes())
        client.send(b'this_is_the_end')
        print("Finshed sending frame")

        response = client.recv(1024).decode()
        print(response)


if __name__ == "__main__":
    # Open socket for communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", 50065))
    server_socket.listen(1)

    time_to_stop = False

    threads = []

    try:
        while True:
            client_socket, addr = server_socket.accept()
            thread = threading.Thread(target=accept_client, args=[client_socket])
            threads.append(thread)
            thread.start()
    except KeyboardInterrupt:
        server_socket.close()
        time_to_stop = True
        print("Waiting for all connections to end")
        for thread in threads:
            thread.join()
        print("Exiting")
