
#!/usr/bin/env python3
import socket, struct, time
import cv2
import numpy as np
from typing import Optional, Tuple, Dict

SERVER_IP = "10.1.38.22"
SERVER_PORT = 5001
HEADER_FMT = ">IQ"         # 4B length, 8B camera timestamp (ns)
# ===== Fast red-ball detector (single image) =====
def detect_red_ball(
    img_bgr: np.ndarray,
    *,
    scale: float = 0.5,
    hsv1: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((0,120,70),(10,255,255)),
    hsv2: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((170,120,70),(180,255,255)),
    min_radius_px: int = 3,
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 5,
    return_mask: bool = False
) -> Optional[Dict]:
    assert img_bgr is not None and img_bgr.ndim == 3, "img_bgr must be a color image"

    if scale != 1.0:
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small = img_bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    (l1, u1), (l2, u2) = hsv1, hsv2
    lower1, upper1 = np.array(l1, np.uint8), np.array(u1, np.uint8)
    lower2, upper2 = np.array(l2, np.uint8), np.array(u2, np.uint8)
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

    result = {
        "center": (float(cx), float(cy)),
        "radius": float(R),
        "contour": (c * inv).astype(np.float32)
    }
    if return_mask:
        if scale != 1.0:
            H, W = img_bgr.shape[:2]
            mask_full = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            mask_full = mask
        result["mask"] = mask_full
    return result

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
    frames, t0 = 0, time.time()
    try:
        while True:
            # Read 4-byte big-endian length, then JPEG payload
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

            t_in = time.time()
            det = detect_red_ball(img, scale=0.5)  # adjust scale for speed vs accuracy
            t_out = time.time()

            if det is not None:
                (cx, cy) = det["center"]
                r = det["radius"]
                cv2.circle(img, (int(cx), int(cy)), int(r), (0,0,255), 2)
                cv2.circle(img, (int(cx), int(cy)), 3, (0,255,0), -1)
                cv2.putText(img, f"({int(cx)},{int(cy)}) r={r:.1f} pix  inf={1000*(t_out-t_in):.1f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                print(f"[Ball] cx={int(cx)} cy={int(cy)} r={r:.1f}")
            else:
                cv2.putText(img, "No red ball", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
                print(f"[Client] ~{fps:.1f} FPS")

            cv2.imshow("Stream+RedBall", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
