#!/usr/bin/env python3
import argparse
import socket
import struct
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Config / formats ----------------
SPORTS_BALL_CLS = 32  # COCO id for "sports ball"
STATE_FMT = "<7d"     # x,y,z,roll,pitch,yaw,gripper (float64)
COORDS_FMT = "<iiiI"  # cx:int32, cy:int32, score_x1000:int32, t_ms:uint32

# --------------- Networking helpers ----------------
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes or None if disconnected."""
    buf = bytearray(n)
    view = memoryview(buf)
    while n:
        chunk = sock.recv(n)
        if not chunk:
            return None
        view[:len(chunk)] = chunk
        view = view[len(chunk):]
        n -= len(chunk)
    return bytes(buf)

class StateReceiver:
    """Non-blocking UDP receiver for Franka state packets <7d>."""
    def __init__(self, ip: str = "0.0.0.0", port: int = 9091):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.0)  # fully non-blocking
        self.size = struct.calcsize(STATE_FMT)
        self.latest: Optional[Tuple[float, ...]] = None

    def poll(self) -> Optional[Tuple[float, ...]]:
        try:
            data, _ = self.sock.recvfrom(2048)
            if len(data) == self.size:
                self.latest = struct.unpack(STATE_FMT, data)
        except BlockingIOError:
            pass
        except Exception:
            pass
        return self.latest

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

# --------------- YOLO + client main ----------------
def main():
    ap = argparse.ArgumentParser("YOLO client: receive JPEG stream, detect ball, listen Franka state")
    # Stream source (Camera PC TCP server)
    ap.add_argument("--server-ip", default="10.1.38.22", help="Camera PC IP")
    ap.add_argument("--server-port", type=int, default=5001, help="Camera TCP port")
    # YOLO
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--device", default="cuda:0", help="'cuda:0' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.15)
    # Franka state (UDP in)
    ap.add_argument("--state-ip", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=9091)
    ap.add_argument("--state-log-every", type=int, default=30, help="print state every N frames")
    # Optional: rebroadcast coords (UDP out)
    ap.add_argument("--coords-ip", default=None, help="If set, send coords packets here")
    ap.add_argument("--coords-port", type=int, default=9092)
    # UI
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # Connect to Camera PC stream
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))
    print(f"[Client] Connected to {args.server_ip}:{args.server_port}")

    # Load YOLO
    model = YOLO(args.model)
    try:
        model.fuse()
    except Exception:
        pass
    try:
        if args.device:
            model.to(args.device)
            if "cuda" in args.device:
                try:
                    # use half if supported
                    model.model.half()
                except Exception:
                    pass
    except Exception:
        pass

    # Franka state receiver (non-blocking)
    state_rx = StateReceiver(args.state_ip, args.state_port)

    # Optional coords UDP sender
    coords_sock = None
    coords_dst = None
    if args.coords_ip:
        coords_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        coords_dst = (args.coords_ip, args.coords_port)
        print(f"[Client] Will send coords to {coords_dst}")

    frames, t0 = 0, time.time()
    try:
        while True:
            # 1) Receive one JPEG frame (4B len prefix, big-endian)
            hdr = recv_all(sock, 4)
            if hdr is None:
                print("[Client] Stream disconnected.")
                break
            (length,) = struct.unpack(">I", hdr)
            jpg = recv_all(sock, length)
            if jpg is None:
                print("[Client] Stream disconnected.")
                break

            # 2) Decode
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 3) YOLO inference (sports ball only)
            t_in = time.time()
            res = model(img, conf=args.conf, iou=0.25, imgsz=args.imgsz,
                        classes=[SPORTS_BALL_CLS], max_det=1, verbose=False)[0]
            t_out = time.time()

            cx, cy, score = -1, -1, 0.0
            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes[0]
                x1, y1, x2, y2 = map(int, map(float, b.xyxy[0]))
                score = float(b.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if args.show:
                    vis = img
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(vis, f"ball {score:.2f}  {1000*(t_out-t_in):.1f}ms",
                                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                if args.show:
                    vis = img

            # 4) Poll Franka state (non-blocking)
            st = state_rx.poll()
            if st is not None and frames % max(1, args.state_log_every) == 0:
                x, y, z, r, p, yw, g = st
                print(f"[State] (x,y,z)=({x:.3f},{y:.3f},{z:.3f}) (r,p,y)=({r:.2f},{p:.2f},{yw:.2f}) g={g:.3f}")

            # 5) Print & (optionally) send coords
            print(f"[Ball] cx={cx} cy={cy} score={score:.3f}")
            if coords_sock and coords_dst:
                t_ms = int(time.time() * 1000) & 0xFFFFFFFF
                pkt = struct.pack(COORDS_FMT, int(cx), int(cy), int(max(0.0, min(1.0, score))*1000), t_ms)
                coords_sock.sendto(pkt, coords_dst)

            # 6) UI
            if args.show:
                cv2.imshow("YOLO Ball", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
                print(f"[Client] ~{fps:.1f} FPS")

    finally:
        try: sock.close()
        except: pass
        state_rx.close()
        if coords_sock:
            try: coords_sock.close()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
