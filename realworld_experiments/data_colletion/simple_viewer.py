#!/usr/bin/env python3
import socket, struct
import cv2, numpy as np

SERVER_IP = "10.1.38.22"   # Camera PC
SERVER_PORT = 5001
HEADER_FMT = ">IQ"         # 4B length, 8B camera timestamp (ns)

def recv_all(sock, n):
    buf = bytearray(n); view = memoryview(buf)
    while n:
        chunk = sock.recv(n)
        if not chunk:
            return None
        view[:len(chunk)] = chunk
        view = view[len(chunk):]
        n -= len(chunk)
    return bytes(buf)

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP, SERVER_PORT))
    print(f"[Client] Connected to {SERVER_IP}:{SERVER_PORT}")

    hdr_size = struct.calcsize(HEADER_FMT)

    try:
        while True:
            hdr = recv_all(s, hdr_size)
            if hdr is None:
                print("[Client] Disconnected.")
                break

            length, t_cam_ns = struct.unpack(HEADER_FMT, hdr)
            jpg = recv_all(s, length)
            if jpg is None:
                print("[Client] Disconnected.")
                break

            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("[Client] decode failed")
                continue

            cv2.putText(img, f"t_cam_ns={t_cam_ns}", (8,24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Stream (timestamped)", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        s.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
