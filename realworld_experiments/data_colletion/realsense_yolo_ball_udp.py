#!/usr/bin/env python3
import socket, struct, time
import cv2
import numpy as np
from ultralytics import YOLO

SERVER_IP = "10.1.38.22"
SERVER_PORT = 5001
SPORTS_BALL_CLS = 32

# Tune for speed
MODEL_PATH = "yolo11n.pt"
IMGSZ = 320
CONF = 0.25
DEVICE = "cuda:0"   # or None / "cpu"
HEADER_FMT = ">IQ" 
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
    # Connect first
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))
    print(f"[Client] Connected to {SERVER_IP}:{SERVER_PORT}")
    hdr_size = struct.calcsize(HEADER_FMT)
    # Load YOLO
    model = YOLO(MODEL_PATH)
    try:
        model.fuse()
    except Exception:
        pass
    try:
        if DEVICE:
            model.to(DEVICE)
            if "cuda" in DEVICE:
                try: model.model.half()
                except Exception: pass
    except Exception: pass

    frames, t0 = 0, time.time()
    try:
        while True:
            header = recv_all(sock, hdr_size)
            if header is None:
                print("[Client] Disconnected")
                break
            (length, t_cam_ns) = struct.unpack(HEADER_FMT, header)
            jpg = recv_all(sock, length)
            if jpg is None:
                print("[Client] Disconnected")
                break

            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # YOLO inference for 'sports ball'
            t_in = time.time()
            res = model(img, conf=CONF, imgsz=IMGSZ,
                        classes=[SPORTS_BALL_CLS], max_det=1, verbose=False)[0]
            t_out = time.time()

            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes[0]
                x1, y1, x2, y2 = map(int, map(float, b.xyxy[0]))
                score = float(b.conf[0])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(img, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(img, f"({cx},{cy}) conf={score:.2f} inf={1000*(t_out-t_in):.1f}ms",
                            (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print(f"[Ball] cx={cx} cy={cy} score={score:.3f}")
            else:
                # no detection
                pass

            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
                print(f"[Client] ~{fps:.1f} FPS")

            cv2.imshow("Stream+YOLO", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
