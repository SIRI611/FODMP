
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Red-Ball client:
- Receives JPEG frames over TCP (4-byte big-endian length prefix)
- Detects a *red* ball using HSV thresholding + morphology + minEnclosingCircle
- Listens to Franka state over UDP (non-blocking)
- (Optional) Re-broadcasts ball coords over UDP as <iiiI>: cx, cy, score*1000, t_ms
- Optional on-screen preview

This is a drop-in replacement for the YOLO version, but much faster with
minimal latency (no neural net). Keep your same server and state inputs.
"""

import argparse
import socket
import struct
import time
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

COORDS_FMT = "<iiiI"   # cx:int32, cy:int32, score_x1000:int32, t_ms:uint32

# --------------------------
# Utilities
# --------------------------
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n)
    view = memoryview(buf)
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        view[:len(chunk)] = chunk
        view = view[len(chunk):]
        remaining -= len(chunk)
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
    """
    Return dict {'center': (cx,cy), 'radius': r, 'mask': mask (optional)} or None if not found.
    """
    assert img_bgr is not None and img_bgr.ndim == 3, "img_bgr must be BGR color"

    # Downscale for speed
    if scale != 1.0:
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = img_bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array(hsv1[0], np.uint8), np.array(hsv1[1], np.uint8)
    lower2, upper2 = np.array(hsv2[0], np.uint8), np.array(hsv2[1], np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    if morph_open_ksize > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((morph_open_ksize,)*2, np.uint8), iterations=1)
    if morph_close_ksize > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((morph_close_ksize,)*2, np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)

    inv = 1.0 / scale
    cx, cy, R = x * inv, y * inv, r * inv
    if R < min_radius_px:
        return None

    return {"center": (float(cx), float(cy)), "radius": float(R)}

class StateReceiver:
    """Non-blocking UDP receiver for Franka state packets. Payload format is user-defined on the sender side."""
    def __init__(self, ip: str = "0.0.0.0", port: int = 9091):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.0)  # fully non-blocking
        self.latest: Optional[bytes] = None

    def poll(self) -> Optional[bytes]:
        try:
            data, _ = self.sock.recvfrom(2048)
            self.latest = data
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

def main():
    ap = argparse.ArgumentParser("Fast Red-Ball client (HSV)")
    # Stream source (Camera PC TCP server)
    ap.add_argument("--server-ip", default="10.1.38.22", help="Camera PC IP")
    ap.add_argument("--server-port", type=int, default=5001, help="Camera TCP port")

    # HSV detector params
    ap.add_argument("--scale", type=float, default=0.5, help="Downscale factor for speed")
    ap.add_argument("--h1l", type=int, default=0);   ap.add_argument("--h1u", type=int, default=10)
    ap.add_argument("--h2l", type=int, default=170); ap.add_argument("--h2u", type=int, default=180)
    ap.add_argument("--smin", type=int, default=120); ap.add_argument("--vmin", type=int, default=70)
    ap.add_argument("--open", type=int, default=3); ap.add_argument("--close", type=int, default=5)
    ap.add_argument("--minr", type=int, default=3, help="Min radius (px) to accept detection")

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

    # Connect to Camera PC stream (4B len prefix, big-endian)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))
    print(f"[Client] Connected to {args.server_ip}:{args.server_port}")

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
            # 1) Receive one JPEG frame
            hdr = recv_all(sock, 4)
            if hdr is None:
                print("[Client] Stream disconnected."); break
            (length,) = struct.unpack(">I", hdr)
            jpg = recv_all(sock, length)
            if jpg is None:
                print("[Client] Stream disconnected."); break

            # 2) Decode
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            vis = img.copy()

            # 3) Red-ball detection
            hsv1 = ((args.h1l, args.smin, args.vmin), (args.h1u, 255, 255))
            hsv2 = ((args.h2l, args.smin, args.vmin), (args.h2u, 255, 255))
            t_in = time.time()
            det = detect_red_ball(
                vis, scale=args.scale, hsv1=hsv1, hsv2=hsv2,
                morph_open_ksize=args.open, morph_close_ksize=args.close, min_radius_px=args.minr
            )
            t_out = time.time()

            cx = cy = -1; score = 1.0  # "score" placeholder for compatibility
            if det is not None:
                (cx, cy) = map(int, det["center"]); r = int(det["radius"])
                if args.show:
                    cv2.circle(vis, (cx, cy), r, (0, 0, 255), 2)
                    cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
                    ms = 1000.0 * (t_out - t_in)
                    cv2.putText(vis, f"red-ball  {ms:.1f}ms", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if args.show:
                    cv2.putText(vis, "no red ball", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 4) Poll Franka state (non-blocking)
            st = state_rx.poll()
            if st is not None and frames % max(1, args.state_log_every) == 0:
                print(f"[State] {len(st)} bytes (user-defined format)")

            # 5) Print & (optionally) send coords
            print(f"[Ball] cx={cx} cy={cy} score={score:.3f}")
            if coords_sock and coords_dst:
                t_ms = int(time.time() * 1000) & 0xFFFFFFFF
                pkt = struct.pack(COORDS_FMT, int(cx), int(cy), int(max(0.0, min(1.0, score)) * 1000), t_ms)
                coords_sock.sendto(pkt, coords_dst)

            # 6) UI
            if args.show:
                cv2.imshow("Red Ball Client", vis)
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
