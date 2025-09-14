#!/usr/bin/env python3
import socket, struct, time
import cv2, numpy as np
import pyrealsense2 as rs

HOST = "0.0.0.0"
PORT = 5001
WIDTH, HEIGHT, FPS = 1280, 720, 30
JPEG_QUALITY = 80

# Header: 4B big-endian length, 8B big-endian camera timestamp (ns)
HEADER_FMT = ">IQ"  # (length:int32, t_cam_ns:uint64)

def start_realsense(width=640, height=480, fps=30):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    prof = pipe.start(cfg)
    return pipe

def send_all(s, b):
    mv = memoryview(b)
    while mv:
        n = s.send(mv)
        mv = mv[n:]

def main():
    pipe = start_realsense(WIDTH, HEIGHT, FPS)
    print(f"[Camera] RealSense {WIDTH}x{HEIGHT}@{FPS}")
    print(f"[Camera] Listening on {HOST}:{PORT}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)

    try:
        while True:
            conn, addr = srv.accept()
            print(f"[Camera] Client {addr} connected")
            t0 = time.time()
            frames = 0
            try:
                while True:
                    frameset = pipe.wait_for_frames()
                    c = frameset.get_color_frame()
                    if not c:
                        continue

                    # Timestamp as close to capture as possible
                    t_cam_ns = time.time_ns()

                    frame = np.asanyarray(c.get_data())  # BGR
                    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if not ok:
                        continue
                    payload = jpg.tobytes()

                    header = struct.pack(HEADER_FMT, len(payload), t_cam_ns)
                    send_all(conn, header)
                    send_all(conn, payload)

                    frames += 1
                    if frames % 60 == 0:
                        fps = frames / (time.time() - t0 + 1e-6)
                        print(f"[Camera] ~{fps:.1f} FPS")
            except (BrokenPipeError, ConnectionResetError):
                print("[Camera] Client disconnected")
            finally:
                try: conn.close()
                except: pass
    finally:
        pipe.stop()

if __name__ == "__main__":
    main()

