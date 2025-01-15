import socket
import threading
from time import sleep


def accept_client(client: socket):
        client.send("This is a test\n".encode())
        i = 0
        while True and not time_to_stop:
            client.send((str(i) + "\n").encode())
            i+=1
            sleep(1)

        client.close()


if __name__ == "__main__":
    # Open socket for communication
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 50067))
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
