import socket
import cv2
import pickle
import struct

HOST = "localhost"
PORT = 12345

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print('Waiting for connection...')
conn, addr = server_socket.accept()
print('Connected by', addr)

cap = cv2.VideoCapture(0) # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Serialize the frame
    data = pickle.dumps(frame)
    # Send the length of the data, followed by the data itself
    message_size = struct.pack("L", len(data)) 
    conn.sendall(message_size + data)

cap.release()
conn.close()
server_socket.close()