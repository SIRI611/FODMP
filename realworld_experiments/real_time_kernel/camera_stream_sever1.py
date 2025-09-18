#!/usr/bin/env python3
"""
Camera Stream Server (TCP)
- Streams RGB frames (JPEG) and optional Depth frames (PNG16) to a single TCP client.
- Header format (big-endian):
  RGB only:  > I Q         -> (rgb_len:int32, t_cam_ns:uint64)
  RGBD:      > I Q I       -> (rgb_len:int32, t_cam_ns:uint64, depth_len:int32)
  Then payload(s): [rgb_bytes] and, if enabled, [depth_png_bytes].

Notes:
- If --enable-depth is set, this tries to use Intel RealSense via pyrealsense2.
- If pyrealsense2 is not available or no RealSense device is found, depth mode will fail.
- If you only need RGB, do NOT pass --enable-depth.
"""
import argparse
import socket
import struct
import sys
import time
import threading

import numpy as np
import cv2

def now_ns() -> int:
    return time.time_ns()

class TCPServer:
    def __init__(self, host: str, port: int, backlog: int = 1):
        self.host = host
        self.port = port
        self.backlog = backlog
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(self.backlog)
        print(f"[Camera] Listening on {self.host}:{self.port}")

    def accept(self):
        conn, addr = self.sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[Camera] Client connected from {addr[0]}:{addr[1]}")
        return conn, addr

def try_import_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except Exception as e:
        return None

class RGBSource:
    """OpenCV VideoCapture as a simple RGB source."""
    def __init__(self, cam_index: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(cam_index)
        if width > 0:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:    self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera index {}".format(cam_index))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        # Ensure BGR -> BGR (OpenCV default). We'll JPEG encode later.
        return frame

    def release(self):
        self.cap.release()

class RealSenseSource:
    """Intel RealSense RGB + Depth using pyrealsense2."""
    def __init__(self, width: int, height: int, fps: int):
        rs = try_import_realsense()
        if rs is None:
            raise RuntimeError("pyrealsense2 not available, cannot enable depth.")
        self.rs = rs
        self.pipe = rs.pipeline()
        self.cfg  = rs.config()
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipe.start(self.cfg)
        # Save intrinsics if desired in the future
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"[Camera] RealSense depth_scale = {self.depth_scale}")

    def read(self):
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError("RealSense frame not ready")
        color_img = np.asanyarray(color.get_data())  # BGR
        depth_img = np.asanyarray(depth.get_data())  # uint16 depth in units
        return color_img, depth_img

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
    return bytes(enc)

def encode_png16(depth_u16: np.ndarray) -> bytes:
    # Ensure uint16
    if depth_u16.dtype != np.uint16:
        depth_u16 = depth_u16.astype(np.uint16, copy=False)
    enc = cv2.imencode(".png", depth_u16)[1]
    return bytes(enc)

def serve_rgb_only(args):
    srv = TCPServer(args.host, args.port)
    conn, _ = srv.accept()
    cap = RGBSource(args.cam, args.width, args.height, args.fps)
    try:
        last_print = 0
        while True:
            frame = cap.read()  # BGR
            t_ns = now_ns()
            jpg = encode_jpeg(frame, quality=args.jpeg_quality)
            header = struct.pack(">IQ", len(jpg), t_ns)
            conn.sendall(header)
            conn.sendall(jpg)
            if time.time() - last_print > 1.0:
                last_print = time.time()
                h, w = frame.shape[:2]
                print(f"[Camera] RGB sent: {w}x{h}, {len(jpg)} bytes")
    except (BrokenPipeError, ConnectionResetError):
        print("[Camera] Client disconnected.")
    finally:
        cap.release()
        conn.close()

def serve_rgbd(args):
    rs = RealSenseSource(args.width, args.height, args.fps)
    srv = TCPServer(args.host, args.port)
    conn, _ = srv.accept()
    try:
        last_print = 0
        while True:
            color, depth = rs.read()
            t_ns = now_ns()
            jpg = encode_jpeg(color, quality=args.jpeg_quality)
            png = encode_png16(depth)
            header = struct.pack(">IQI", len(jpg), t_ns, len(png))
            conn.sendall(header)
            conn.sendall(jpg)
            conn.sendall(png)
            if time.time() - last_print > 1.0:
                last_print = time.time()
                h, w = color.shape[:2]
                print(f"[Camera] RGBD sent: {w}x{h}, RGB {len(jpg)} bytes, Depth {len(png)} bytes")
    except (BrokenPipeError, ConnectionResetError):
        print("[Camera] Client disconnected.")
    finally:
        rs.stop()
        conn.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--enable-depth", action="store_true", help="Enable RealSense RGBD streaming")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index for RGB-only mode")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    args = ap.parse_args()

    if args.enable_depth:
        serve_rgbd(args)
    else:
        serve_rgb_only(args)

if __name__ == "__main__":
    main()
