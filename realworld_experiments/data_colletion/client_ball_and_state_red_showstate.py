#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Red-Ball client with Franka state overlay.

- Receives JPEG frames over TCP (4-byte big-endian length prefix)
- Detects a red ball via HSV (dual-range) + morphology + minEnclosingCircle
- Receives Franka telemetry over UDP and PARSES it (basic 7d or full 27d)
- Overlays pose (and optionally twist) on the video
- Optionally re-broadcasts detected pixel coords via UDP

Telemetry formats (little-endian) must match the server:
    basic: <7d -> [x, y, z, roll, pitch, yaw, gripper_ratio]
    full:  <27d -> [x y z r p y] + [vx vy vz wx wy wz] + [q1..q7] + [dq1..dq7] + [g]
"""

import argparse
import socket
import struct
import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

COORDS_FMT = "<iiiI"   # cx, cy, score*1000, t_ms
HEADER_FMT = ">IQ"     # (length:int32, t_cam_ns:uint64)
# --------------------------
# Helpers
# --------------------------
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n); view = memoryview(buf); rem = n
    while rem:
        chunk = sock.recv(rem)
        if not chunk: return None
        view[:len(chunk)] = chunk; view = view[len(chunk):]; rem -= len(chunk)
    return bytes(buf)

def detect_red_ball(
    img_bgr: np.ndarray,
    *,
    scale: float = 0.5,
    hsv1: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((0,120,70),(10,255,255)),
    hsv2: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((170,120,70),(180,255,255)),
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 5,
    min_radius_px: int = 3,
) -> Optional[Dict]:
    if img_bgr is None or img_bgr.ndim != 3: return None
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

class FrankaStateRX:
    """
    UDP listener that parses telemetry packets.
    """
    def __init__(self, ip: str, port: int, mode: str = "basic"):
        assert mode in ("basic","full")
        self.addr = (ip, port)
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.0)
        # Precompute sizes
        self._fmt_basic = "<7d"; self._len_basic = struct.calcsize(self._fmt_basic)
        self._fmt_full  = "<27d"; self._len_full  = struct.calcsize(self._fmt_full)
        self.latest = None  # dict

    def poll(self) -> Optional[Dict]:
        try:
            data, _ = self.sock.recvfrom(4096)
        except BlockingIOError:
            return self.latest
        except Exception:
            return self.latest

        if self.mode == "basic" and len(data) >= self._len_basic:
            vals = struct.unpack(self._fmt_basic, data[:self._len_basic])
            x,y,z,r,p,yaw,g = vals
            self.latest = {"x":x,"y":y,"z":z,"roll":r,"pitch":p,"yaw":yaw,"g":g}
        elif self.mode == "full" and len(data) >= self._len_full:
            vals = struct.unpack(self._fmt_full, data[:self._len_full])
            x,y,z,r,p,yaw = vals[0:6]
            vx,vy,vz,wx,wy,wz = vals[6:12]
            q = vals[12:19]; dq = vals[19:26]; g = vals[26]
            self.latest = {"x":x,"y":y,"z":z,"roll":r,"pitch":p,"yaw":yaw,
                           "vx":vx,"vy":vy,"vz":vz,"wx":wx,"wy":wy,"wz":wz,
                           "q":q,"dq":dq,"g":g}
        return self.latest

    def close(self):
        try: self.sock.close()
        except: pass

def overlay_state(img, st: Dict, mode: str, org=(8,18)):
    line = lambda i: (org[0], org[1] + 18*i)
    if st is None: return img
    vis = img
    if mode == "basic":
        text1 = f"EE xyz=({st['x']:+.3f},{st['y']:+.3f},{st['z']:+.3f}) m"
        text2 = f"EE rpy=({st['roll']:+.2f},{st['pitch']:+.2f},{st['yaw']:+.2f}) rad  g={st['g']:.2f}"
        cv2.putText(vis, text1, line(0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(vis, text2, line(1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 2)
    else:
        text1 = f"EE xyz=({st['x']:+.3f},{st['y']:+.3f},{st['z']:+.3f})  rpy=({st['roll']:+.2f},{st['pitch']:+.2f},{st['yaw']:+.2f})"
        text2 = f"Twist v=({st['vx']:+.2f},{st['vy']:+.2f},{st['vz']:+.2f})  w=({st['wx']:+.2f},{st['wy']:+.2f},{st['wz']:+.2f})"
        cv2.putText(vis, text1, line(0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(vis, text2, line(1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 2)
    return vis

def main():
    ap = argparse.ArgumentParser("Red-ball client + Franka state overlay")
    # Stream source
    ap.add_argument("--server-ip", default="10.1.38.22")
    ap.add_argument("--server-port", type=int, default=5001)

    # Franka state (UDP in)
    ap.add_argument("--state-ip", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=9091)
    ap.add_argument("--state-mode", choices=["basic","full"], default="basic",
                    help="Match the sender (--telemetry on the robot server).")

    # HSV
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--h1l", type=int, default=0);   ap.add_argument("--h1u", type=int, default=10)
    ap.add_argument("--h2l", type=int, default=170); ap.add_argument("--h2u", type=int, default=180)
    ap.add_argument("--smin", type=int, default=120); ap.add_argument("--vmin", type=int, default=70)
    ap.add_argument("--open", type=int, default=3); ap.add_argument("--close", type=int, default=5)
    ap.add_argument("--minr", type=int, default=3)

    # Optional rebroadcast
    ap.add_argument("--coords-ip", default=None)
    ap.add_argument("--coords-port", type=int, default=9092)

    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # Connect TCP
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))
    print(f"[Client] Connected to {args.server_ip}:{args.server_port}")

    # Franka state receiver
    st_rx = FrankaStateRX(args.state_ip, args.state_port, args.state_mode)
    hdr_size = struct.calcsize(HEADER_FMT)
    # Optional coords UDP
    coords_sock = None; coords_dst = None
    if args.coords_ip:
        coords_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        coords_dst = (args.coords_ip, args.coords_port)
        print(f"[Client] Will send coords to {coords_dst}")

    frames, t0 = 0, time.time()
    try:
        while True:
            # Frame
            hdr = recv_all(sock, hdr_size)
            if hdr is None: print("[Client] Stream disconnected."); break
            (length, t_cam_ns) = struct.unpack(HEADER_FMT, hdr)
            jpg = recv_all(sock, length)
            if jpg is None: print("[Client] Stream disconnected."); break
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue

            # Detect
            hsv1 = ((args.h1l, args.smin, args.vmin), (args.h1u, 255, 255))
            hsv2 = ((args.h2l, args.smin, args.vmin), (args.h2u, 255, 255))
            t_in = time.time()
            det = detect_red_ball(img, scale=args.scale, hsv1=hsv1, hsv2=hsv2,
                                  morph_open_ksize=args.open, morph_close_ksize=args.close, min_radius_px=args.minr)
            t_out = time.time()

            cx = cy = -1; score = 1.0
            if det is not None:
                cx, cy = map(int, det["center"]); r = int(det["radius"])
                if args.show:
                    cv2.circle(img, (cx, cy), r, (0,0,255), 2)
                    cv2.circle(img, (cx, cy), 3, (0,255,0), -1)
                    cv2.putText(img, f"red-ball {(t_out-t_in)*1000:.1f}ms", (8, img.shape[0]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            print(f"[Ball] cx={cx} cy={cy} r={r if det else -1}") 
            # State
            st = st_rx.poll()
            print(f"[State] {st}")
            if args.show:
                img = overlay_state(img, st, args.state_mode)

            # Coords UDP
            if coords_sock and coords_dst:
                t_ms = int(time.time() * 1000) & 0xFFFFFFFF
                pkt = struct.pack(COORDS_FMT, int(cx), int(cy), int(max(0.0, min(1.0, score))*1000), t_ms)
                coords_sock.sendto(pkt, coords_dst)

            if args.show:
                cv2.imshow("Client (HSV + Franka state)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
                print(f"[Client] ~{fps:.1f} FPS")

    finally:
        try: sock.close()
        except: pass
        st_rx.close()
        if coords_sock:
            try: coords_sock.close()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
