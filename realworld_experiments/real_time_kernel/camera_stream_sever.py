#!/usr/bin/env python3
import socket, struct, time
import cv2, numpy as np
import pyrealsense2 as rs

HOST = "0.0.0.0"
PORT = 5001
WIDTH, HEIGHT, FPS = 1280, 720, 30
JPEG_QUALITY = 80

# Header formats
HEADER_RGB_FMT = ">IQ"     # RGB only: (length:int32, t_cam_ns:uint64)
HEADER_RGBD_FMT = ">IQI"   # RGB+Depth: (rgb_length:int32, t_cam_ns:uint64, depth_length:int32)

def start_realsense(width=WIDTH, height=HEIGHT, fps=30, enable_depth=True):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if enable_depth:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    prof = pipe.start(cfg)
    return pipe

def send_all(s, b):
    mv = memoryview(b)
    while mv:
        n = s.send(mv)
        mv = mv[n:]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-depth", action="store_true", help="Send depth data alongside RGB")
    args = ap.parse_args()
    
    pipe = start_realsense(WIDTH, HEIGHT, FPS, enable_depth=args.enable_depth)
    stream_type = "RGB+Depth" if args.enable_depth else "RGB only"
    print(f"[Camera] RealSense {WIDTH}x{HEIGHT}@{FPS} ({stream_type})")
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
                    print("[Server] Waiting for frames...")
                    frameset = pipe.wait_for_frames()
                    print("[Server] Got frameset")
                    c = frameset.get_color_frame()
                    if not c:
                        print("[Server] No color frame, continuing...")
                        continue
                    print("[Server] Got color frame")

                    # Timestamp as close to capture as possible
                    t_cam_ns = time.time_ns()

                    # RGB data
                    frame = np.asanyarray(c.get_data())  # BGR
                    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if not ok:
                        continue
                    rgb_payload = jpg.tobytes()

                    if args.enable_depth:
                        # Depth data
                        d = frameset.get_depth_frame()
                        if d:
                            print('[Server] Got depth frame')
                            depth_frame = np.asanyarray(d.get_data(), dtype=np.uint16)  # 16-bit depth in mm
                            depth_payload = depth_frame.tobytes()
                            
                            # Send RGB+Depth header and data
                            header = struct.pack(HEADER_RGBD_FMT, len(rgb_payload), t_cam_ns, len(depth_payload))
                            send_all(conn, header)
                            send_all(conn, rgb_payload)
                            send_all(conn, depth_payload)
                        else:
                            # Fallback to RGB only if no depth
                            header = struct.pack(HEADER_RGB_FMT, len(rgb_payload), t_cam_ns)
                            send_all(conn, header)
                            send_all(conn, rgb_payload)
                    else:
                        # RGB only
                        header = struct.pack(HEADER_RGB_FMT, len(rgb_payload), t_cam_ns)
                        print(f"[Server] Sending RGB frame: {len(rgb_payload)} bytes")
                        send_all(conn, header)
                        send_all(conn, rgb_payload)
                        print(f"[Server] Sent RGB frame")

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

