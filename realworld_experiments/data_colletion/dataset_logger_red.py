#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset logger (HSV red-ball detection + Franka state sync + CSV output).
- Receives JPEG frames over TCP with 12-byte header (>IQ): (length:int32, t_cam_ns:uint64)
- Detects a red ball via HSV dual-range thresholding + morphology + minEnclosingCircle
- Computes pixel velocity from camera timestamps
- Listens to Franka UDP state packets (<7d: [x,y,z,roll,pitch,yaw,gripper])
- Syncs each frame to nearest Franka state by timestamp (tolerance configurable)
- Writes everything to CSV
"""

import argparse, csv, socket, struct, threading, time
from collections import deque
from typing import Optional, Tuple, Dict

import cv2, numpy as np

STATE_FMT = "<7d"
STATE_SIZE = struct.calcsize(STATE_FMT)
HEADER_FMT = ">IQ"  # (length:int32, t_cam_ns:uint64)

# ---------------- HSV red ball detector ----------------
def detect_red_ball(
    img_bgr: np.ndarray,
    *,
    scale: float = 0.5,
    hsv1=((0,120,70),(10,255,255)),
    hsv2=((170,120,70),(180,255,255)),
    morph_open_ksize=3,
    morph_close_ksize=5,
    min_radius_px=3
) -> Optional[Dict]:
    if img_bgr is None or img_bgr.ndim != 3: return None
    # Optional downscale
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img_bgr
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array(hsv1[0], np.uint8), np.array(hsv1[1], np.uint8)
    lower2, upper2 = np.array(hsv2[0], np.uint8), np.array(hsv2[1], np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    if morph_open_ksize>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((morph_open_ksize,)*2, np.uint8), iterations=1)
    if morph_close_ksize>0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((morph_close_ksize,)*2, np.uint8), iterations=1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea); (x,y), r = cv2.minEnclosingCircle(c)
    inv = 1.0/scale; cx, cy, R = x*inv, y*inv, r*inv
    if R < min_radius_px: return None
    return {"center": (float(cx), float(cy)), "radius": float(R)}

# ---------------- Socket helpers ----------------
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n); view = memoryview(buf)
    while n:
        chunk = sock.recv(n)
        if not chunk: return None
        view[:len(chunk)] = chunk; view = view[len(chunk):]; n -= len(chunk)
    return bytes(buf)

# ---------------- State buffer ----------------
class StateBuffer:
    def __init__(self, maxlen=500):
        self.buf = deque(maxlen=maxlen); self.lock = threading.Lock()
    def push(self, t_ns: int, state: Tuple[float,...]):
        with self.lock: self.buf.append((t_ns, state))
    def get_nearest(self, t_ns: int, max_dt_ns: int):
        with self.lock:
            if not self.buf: return None
            best, best_abs = None, None
            for ts, st in self.buf:
                dt = abs(ts - t_ns)
                if best_abs is None or dt < best_abs:
                    best_abs, best = dt, (ts, st)
            return best if best and best_abs <= max_dt_ns else None

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
                data,_ = self.sock.recvfrom(2048)
                if len(data) == STATE_SIZE:
                    st = struct.unpack(STATE_FMT, data)
                    t_ns = time.time_ns()  # receive timestamp
                    self.buf.push(t_ns, st)
            except Exception:
                time.sleep(0.001)
    def stop(self):
        self._run = False
        try: self.sock.close()
        except: pass

# ---------------- Frame reader ----------------
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
            with self.lock: self.q.append((t_cam_ns, t_recv_ns, img))
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

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser("Dataset logger (HSV red ball + camera timestamp)")
    ap.add_argument("--server-ip", default="10.1.38.22")
    ap.add_argument("--server-port", type=int, default=5001)
    ap.add_argument("--state-ip", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=9091)
    ap.add_argument("--state-sync-ms", type=int, default=40)
    # HSV tuning
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--h1l", type=int, default=0);   ap.add_argument("--h1u", type=int, default=10)
    ap.add_argument("--h2l", type=int, default=170); ap.add_argument("--h2u", type=int, default=180)
    ap.add_argument("--smin", type=int, default=120); ap.add_argument("--vmin", type=int, default=70)
    ap.add_argument("--open", type=int, default=3); ap.add_argument("--close", type=int, default=5)
    ap.add_argument("--minr", type=int, default=3)
    ap.add_argument("--csv", default="ball_dataset.csv")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    fr = FrameReader(args.server_ip, args.server_port); fr.start()
    if not fr.wait_connected(5.0): raise SystemExit("Could not connect to Camera PC stream.")
    state_buf = StateBuffer(500); sr = StateReceiver(args.state_ip, args.state_port, state_buf); sr.start()
    hdr_size = struct.calcsize(HEADER_FMT)
    f = open(args.csv, "w", newline=""); w = csv.writer(f)
    w.writerow([
        "t_cam_ns","t_recv_ns","frame_id","cx","cy","vx","vy","conf",
        "x","y","z","roll","pitch","yaw","gripper","cartesian_v","joint_v"
    ]); f.flush()

    prev = None; frame_id = 0; max_dt_ns = args.state_sync_ms * 1_000_000
    print(f"[INFO] Logging to {args.csv}. Press 'q' to stop.")

    try:
        while True:
            item = fr.get_latest()
            if item is None:
                time.sleep(0.001); continue
            t_cam_ns, t_recv_ns, img = item; frame_id += 1

            hsv1 = ((args.h1l, args.smin, args.vmin), (args.h1u, 255, 255))
            hsv2 = ((args.h2l, args.smin, args.vmin), (args.h2u, 255, 255))
            det = detect_red_ball(
                img, scale=args.scale, hsv1=hsv1, hsv2=hsv2,
                morph_open_ksize=args.open, morph_close_ksize=args.close, min_radius_px=args.minr
            )

            cx = cy = -1; conf = 1.0
            if det: (cx, cy) = map(int, det["center"]); r = int(det["radius"])
            vx = vy = 0.0
            if prev and cx >= 0 and cy >= 0:
                t_prev_ns, cxp, cyp = prev; dt_s = (t_cam_ns - t_prev_ns) / 1e9
                if dt_s > 0: vx = (cx - cxp) / dt_s; vy = (cy - cyp) / dt_s
            if cx >= 0 and cy >= 0: prev = (t_cam_ns, cx, cy)

            match = state_buf.get_nearest(t_cam_ns, max_dt_ns)
            if match: _, st = match; x,y,z,rll,pch,yw,g = st[:7]; cartesian_v = joint_v = float("nan")
            else: x=y=z=rll=pch=yw=g=cartesian_v=joint_v=float("nan")

            w.writerow([t_cam_ns, t_recv_ns, frame_id, cx, cy, vx, vy, conf,
                        x, y, z, rll, pch, yw, g, cartesian_v, joint_v])
            f.flush()

            if args.show:
                vis = img.copy()
                if cx>=0 and cy>=0:
                    cv2.circle(vis, (cx, cy), int(r), (0,0,255), 2)
                    cv2.circle(vis, (cx, cy), 3, (0,255,0), -1)
                cv2.putText(vis, f"frame={frame_id}", (8,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Dataset Logger (HSV)", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        sr.stop(); fr.stop(); f.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
