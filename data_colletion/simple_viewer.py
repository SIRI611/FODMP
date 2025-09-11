#!/usr/bin/env python3
import socket, struct
import cv2
import numpy as np

SERVER_IP = "10.1.38.22"   # Camera PC IP
SERVER_PORT = 5001

def recv_all(sock, n):
    data = bytearray(n)
    view = memoryview(data)
    while n:
        r = sock.recv(n)
        if not r:
            return None
        view[:len(r)] = r
        view = view[len(r):]
        n -= len(r)
    return data

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    print(f"[Client] Connected to {SERVER_IP}:{SERVER_PORT}")

    try:
        while True:
            # Read 4-byte length
            header = recv_all(sock, 4)
            if header is None:
                print("[Client] Disconnected")
                break
            (length,) = struct.unpack(">I", header)
            # Read JPEG
            jpg = recv_all(sock, length)
            if jpg is None:
                print("[Client] Disconnected")
                break

            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            cv2.imshow("Stream", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
