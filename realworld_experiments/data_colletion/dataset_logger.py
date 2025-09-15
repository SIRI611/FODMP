#!/usr/bin/env python3
import argparse, csv, socket, struct, threading, time
from collections import deque
from typing import Optional, Tuple

import cv2, numpy as np
from ultralytics import YOLO

SPORTS_BALL_CLS = 32
STATE_FMT = "<7d"; STATE_SIZE = struct.calcsize(STATE_FMT)
HEADER_FMT = ">IQ"  # (length:int32, t_cam_ns:uint64)

def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n); view = memoryview(buf)
    while n:
        chunk = sock.recv(n)
        if not chunk: return None
        view[:len(chunk)] = chunk; view = view[len(chunk):]; n -= len(chunk)
    return bytes(buf)

class StateBuffer:
    def __init__(self, maxlen=500):
        self.buf = deque(maxlen=maxlen); self.lock = threading.Lock()
    def push(self, t_ns: int, state: Tuple[float,...]):
        with self.lock: self.buf.append((t_ns, state))
    def get_nearest(self, t_ns: int, max_dt_ns: int):
        with self.lock:
            if not self.buf: return None
            best = None; best_abs = None
            for ts, st in self.buf:
                dt = abs(ts - t_ns)
                if best_abs is None or dt < best_abs:
                    best_abs = dt; best = (ts, st)
            return best if best is not None and best_abs <= max_dt_ns else None

class StateReceiver(threading.Thread):
    def __init__(self, ip: str, port: int, buf: StateBuffer):
        super().__init__(daemon=True)
        self.buf = buf; self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr); self._run = True
    def run(self):
        while self._run:
            try:
                data, _ = self.sock.recvfrom(2048)
                if len(data) == STATE_SIZE:
                    st = struct.unpack(STATE_FMT, data)
                    t_ns = time.time_ns()  # receive timestamp (can switch to robot-provided if available)
                    self.buf.push(t_ns, st)
            except Exception:
                time.sleep(0.001)
    def stop(self):
        self._run = False
        try: self.sock.close()
        except: pass

class FrameReader(threading.Thread):
    """Reads frames with 12B header (>IQ) then JPEG."""
    def __init__(self, server_ip: str, server_port: int, qmax=4):
        super().__init__(daemon=True)
        self.server = (server_ip, server_port)
        self.sock = None; self.q = deque(maxlen=qmax)
        self.lock = threading.Lock(); self._run = True; self._connected = threading.Event()
    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server); self._connected.set()
        while self._run:
            hdr = recv_all(self.sock, struct.calcsize(HEADER_FMT))
            if hdr is None: break
            length, t_cam_ns = struct.unpack(HEADER_FMT, hdr)
            jpg = recv_all(self.sock, length)
            if jpg is None: break
            t_recv_ns = time.time_ns()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            with self.lock:
                self.q.append((t_cam_ns, t_recv_ns, img))
        self._connected.clear()
    def get_latest(self):
        with self.lock:
            if not self.q: return None
            item = self.q.pop(); self.q.clear(); return item
    def wait_connected(self, timeout=5.0): return self._connected.wait(timeout)
    def stop(self):
        self._run = False
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR); self.sock.close()
        except: pass

def main():
    ap = argparse.ArgumentParser("Dataset logger with camera timestamp")
    ap.add_argument("--server-ip", default="10.1.38.22")
    ap.add_argument("--server-port", type=int, default=5001)
    ap.add_argument("--state-ip", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=9091)
    ap.add_argument("--state-sync-ms", type=int, default=40)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--csv", default="ball_dataset.csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fr = FrameReader(args.server_ip, args.server_port); fr.start()
    if not fr.wait_connected(5.0): raise SystemExit("Could not connect to Camera PC stream.")
    state_buf = StateBuffer(500); sr = StateReceiver(args.state_ip, args.state_port, state_buf); sr.start()

    model = YOLO(args.model)
    try: model.fuse()
    except: pass
    try:
        if args.device:
            model.to(args.device)
            if "cuda" in args.device:
                try: model.model.half()
                except: pass
    except: pass

    f = open(args.csv, "w", newline=""); w = csv.writer(f)
    w.writerow([
        "t_cam_ns","t_recv_ns","frame_id","cx","cy","vx","vy","conf",
        "x","y","z","roll","pitch","yaw","gripper", "cartesian_v", "joint_v"
    ]); f.flush()

    prev = None  # (t_cam_ns, cx, cy)
    frame_id = 0; max_dt_ns = args.state_sync_ms * 1_000_000

    print(f"[INFO] Logging to {args.csv}. Press 'q' to stop.")
    try:
        while True:
            item = fr.get_latest()
            if item is None:
                time.sleep(0.001); continue
            t_cam_ns, t_recv_ns, img = item
            frame_id += 1

            t0 = time.time_ns()
            res = model(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                        classes=[SPORTS_BALL_CLS], max_det=1, verbose=False)[0]
            t1 = time.time_ns()

            cx = cy = -1; conf = 0.0
            if res.boxes is not None and len(res.boxes) > 0:
                b = res.boxes[0]
                x1, y1, x2, y2 = map(int, map(float, b.xyxy[0]))
                conf = float(b.conf[0]); cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if args.show:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.circle(img, (cx,cy), 4, (0,0,255), -1)
                    cv2.putText(img, f"conf={conf:.2f}  inf={(t1-t0)/1e6:.1f}ms",
                                (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # velocity in px/s using camera timestamps
            vx = vy = 0.0
            if prev is not None and cx >= 0 and cy >= 0:
                t_prev_ns, cxp, cyp = prev
                dt_s = (t_cam_ns - t_prev_ns) / 1e9
                if dt_s > 0:
                    vx = (cx - cxp) / dt_s; vy = (cy - cyp) / dt_s
            if cx >= 0 and cy >= 0:
                prev = (t_cam_ns, cx, cy)

            # match robot state by camera timestamp
            match = state_buf.get_nearest(t_cam_ns, max_dt_ns)
            if match is not None:
                _, st = match
                x, y, z, r, p, yw, g, cartesian_v, joint_v = st
            else:
                x = y = z = r = p = yw = g = float("nan")
                cartesian_v = joint_v = float("nan")

            w.writerow([t_cam_ns, t_recv_ns, frame_id, cx, cy, vx, vy, conf, x, y, z, r, p, yw, g, cartesian_v, joint_v])
            f.flush()

            if args.show:
                cv2.putText(img, f"frame={frame_id}  sync<={args.state_sync_ms}ms",
                            (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Dataset Logger", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sr.stop(); fr.stop(); f.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
