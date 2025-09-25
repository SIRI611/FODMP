#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Red-Ball client with Franka state overlay + Dataset Logger.

- Receives JPEG frames over TCP (4/16-byte big-endian headers; see HEADER_* below)
- Detects a red ball via HSV (dual-range) + morphology + minEnclosingCircle
- Receives Franka telemetry over UDP and PARSES it (basic 7d or full 27d)
- Overlays pose (and optionally twist) on the video
- Optionally re-broadcasts detected pixel coords via UDP
- Logs a CSV dataset when --log is enabled

Telemetry formats (little-endian) must match the server:
    basic: <7d  -> [x, y, z, roll, pitch, yaw, gripper_ratio]
    full:  <27d -> [x y z r p y] + [vx vy vz wx wy wz] + [q1..q7] + [dq1..dq7] + [g]
"""

import argparse
import csv
import math
import os
import socket
import struct
import time
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

# SciPy (optional for rotations)
try:
    from scipy.spatial.transform import Rotation as SciPyRotation  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Franka (optional; we mainly use UDP telemetry, but can fall back to franky reads)
try:
    from franky import Robot, Gripper  # type: ignore
    import franky  # type: ignore
    _HAS_FRANKY = True
except Exception:
    _HAS_FRANKY = False


# --------------------------
# Constants / Protocol
# --------------------------
GRIP_SPEED = 0.02      # m/s (adjust to your hardware)
GRIP_FORCE = 20.0      # N (adjust to your hardware)
GRIPPER_COOLDOWN = 0.5
GRIPPER_MAX_OPEN = 0.08
GRIPPER_MIN_OPEN = 0.00

COORDS_FMT = "<iiiI"       # cx, cy, score*1000, t_ms
HEADER_RGB_FMT = ">IQ"     # RGB only: (rgb_length:int32, t_cam_ns:uint64)
HEADER_RGBD_FMT = ">IQI"   # RGB+Depth: (rgb_length:int32, t_cam_ns:uint64, depth_length:int32)

# If you also want to read robot directly from this client (not recommended if another machine handles FCI):
ROBOT_IP = "10.1.38.23"

# Camera->World homogeneous transform (4x4). Adjust to your calibration.
t_matrix = np.array([[-0.031968, -0.998314,  0.048438, 1.113756],
                     [ 0.034582,  0.047329,  0.998281, 0.119144],
                     [-0.998890,  0.033588,  0.033010, 1.290525],
                     [ 0.000000,  0.000000,  0.000000, 1.000000]], dtype=np.float64)
transformation = t_matrix


# --------------------------
# Helpers
# --------------------------
def recv_all(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray(n)
    view = memoryview(buf)
    rem = n
    while rem:
        chunk = sock.recv(rem)
        if not chunk:
            return None
        view[:len(chunk)] = chunk
        view = view[len(chunk):]
        rem -= len(chunk)
    return bytes(buf)


def detect_red_ball(
    img_bgr: np.ndarray,
    *,
    scale: float = 0.5,
    hsv1: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((0, 120, 70), (10, 255, 255)),
    hsv2: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = ((170, 120, 70), (180, 255, 255)),
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 5,
    min_radius_px: int = 3,
) -> Optional[Dict]:
    if img_bgr is None or img_bgr.ndim != 3:
        return None
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else img_bgr
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


def pixel_to_camera_coords(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    """Convert pixel (u,v) at depth (m) into camera-frame 3D (x,y,z) meters."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return (float(x), float(y), float(z))


def estimate_depth_from_ball_size(radius_px: float, ball_diameter_m: float, fx: float) -> float:
    """Estimate depth from apparent radius and known physical diameter."""
    if radius_px <= 0:
        return 1.0
    real_radius_m = ball_diameter_m * 0.5
    depth = (real_radius_m * fx) / radius_px
    if depth < 0.1 or depth > 10.0:
        print(f"[Warn] Unusual depth estimate: {depth:.3f} m (radius_px={radius_px:.1f})")
    return float(depth)


# --------------------------
# UDP state receiver
# --------------------------
class FrankaStateRX:
    """UDP listener that parses telemetry packets."""
    def __init__(self, ip: str, port: int, mode: str = "basic"):
        assert mode in ("basic", "full")
        self.addr = (ip, port)
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.addr)
        self.sock.settimeout(0.0)
        self._fmt_basic = "<7d"; self._len_basic = struct.calcsize(self._fmt_basic)
        self._fmt_full  = "<27d"; self._len_full  = struct.calcsize(self._fmt_full)
        self.latest: Optional[Dict] = None

    def poll(self) -> Optional[Dict]:
        try:
            while True:
                data, _ = self.sock.recvfrom(4096)
                if self.mode == "basic" and len(data) >= self._len_basic:
                    x, y, z, r, p, yaw, g = struct.unpack(self._fmt_basic, data[:self._len_basic])
                    self.latest = {"x": x, "y": y, "z": z, "roll": r, "pitch": p, "yaw": yaw, "g": g}
                elif self.mode == "full" and len(data) >= self._len_full:
                    vals = list(struct.unpack(self._fmt_full, data[:self._len_full]))
                    x, y, z, r, p, yaw = vals[0:6]
                    vx, vy, vz, wx, wy, wz = vals[6:12]
                    q = vals[12:19]; dq = vals[19:26]; g = vals[26]
                    self.latest = {
                        "x": x, "y": y, "z": z, "roll": r, "pitch": p, "yaw": yaw,
                        "vx": vx, "vy": vy, "vz": vz, "wx": wx, "wy": wy, "wz": wz,
                        "q": q, "dq": dq, "g": g
                    }
                else:
                    break
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


# --------------------------
# Visualization
# --------------------------
def overlay_state(img, st: Optional[Dict], mode: str, org=(8, 18)):
    if st is None:
        return img
    line = lambda i: (org[0], org[1] + 18 * i)
    vis = img
    if mode == "basic":
        text1 = f"EE xyz=({st['x']:+.3f},{st['y']:+.3f},{st['z']:+.3f}) m"
        text2 = f"EE rpy=({st['roll']:+.2f},{st['pitch']:+.2f},{st['yaw']:+.2f}) rad  g={st['g']:.2f}"
        cv2.putText(vis, text1, line(0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, text2, line(1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
    else:
        text1 = f"EE xyz=({st['x']:+.3f},{st['y']:+.3f},{st['z']:+.3f})  rpy=({st['roll']:+.2f},{st['pitch']:+.2f},{st['yaw']:+.2f})"
        text2 = f"Twist v=({st['vx']:+.2f},{st['vy']:+.2f},{st['vz']:+.2f})  w=({st['wx']:+.2f},{st['wy']:+.2f},{st['wz']:+.2f})"
        cv2.putText(vis, text1, line(0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, text2, line(1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 2)
    return vis


# --------------------------
# Minimal quat helpers (fallback)
# --------------------------
def quat_to_rpy_wxyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to (roll, pitch, yaw), XYZ intrinsic."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else (-1.0 if t2 < -1.0 else t2)
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw


def small_euler_xyz_to_quat_wxyz(rx: float, ry: float, rz: float) -> List[float]:
    hr, hp, hy = rx * 0.5, ry * 0.5, rz * 0.5
    cr, sr = math.cos(hr), math.sin(hr)
    cp, sp = math.cos(hp), math.sin(hp)
    cy, sy = math.cos(hy), math.sin(hy)
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return [w, x, y, z]


def delta_quat_wxyz_from_euler_xyz(droll: float, dpitch: float, dyaw: float) -> List[float]:
    if _HAS_SCIPY:
        qxyzw = SciPyRotation.from_euler("xyz", [droll, dpitch, dyaw]).as_quat()  # [x, y, z, w]
        return [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]
    return small_euler_xyz_to_quat_wxyz(droll, dpitch, dyaw)


# --------------------------
# Optional direct franky reads (fallback only)
# --------------------------
def read_ee_pose_rpy(robot: "Robot") -> Tuple[float, float, float, float, float, float]:
    cart = robot.current_cartesian_state
    pose = cart.pose.end_effector_pose
    x, y, z = pose.translation  # meters
    w, xq, yq, zq = pose.quaternion  # (w, x, y, z)
    r, p, yaw = quat_to_rpy_wxyz(w, xq, yq, zq)
    return float(x), float(y), float(z), float(r), float(p), float(yaw)


def read_gripper_open_ratio(gripper: "franky.Gripper") -> float:
    try:
        width = float(gripper.width)
        width = max(GRIPPER_MIN_OPEN, min(GRIPPER_MAX_OPEN, width))
        closed_ratio = 1.0 - (width - GRIPPER_MIN_OPEN) / (GRIPPER_MAX_OPEN - GRIPPER_MIN_OPEN)
        return float(closed_ratio)
    except Exception:
        return float("nan")


def read_cartesian_twist(robot: "Robot") -> List[float]:
    vel_obj = robot.current_cartesian_state.velocity
    tw = getattr(vel_obj, "end_effector_twist", None) or getattr(vel_obj, "twist", None) or vel_obj
    for lin_name, ang_name in [("linear", "angular"), ("translational", "rotational"), ("v", "w"),
                               ("linear_velocity", "angular_velocity")]:
        lin = getattr(tw, lin_name, None)
        ang = getattr(tw, ang_name, None)
        if lin is not None and ang is not None:
            lin = list(lin) if hasattr(lin, "__iter__") else [lin.x, lin.y, lin.z]
            ang = list(ang) if hasattr(ang, "__iter__") else [ang.x, ang.y, ang.z]
            return [float(l) for l in lin[:3]] + [float(a) for a in ang[:3]]
    comp_names = ("vx", "vy", "vz", "wx", "wy", "wz")
    if all(hasattr(tw, n) for n in comp_names):
        return [float(getattr(tw, n)) for n in comp_names]
    try:
        seq = list(tw)
        if len(seq) == 6:
            return [float(v) for v in seq]
    except TypeError:
        pass
    raise TypeError(f"Unsupported Twist schema: has {dir(tw)}")


def read_joint_position(robot: "Robot") -> List[float]:
    return [float(q) for q in list(robot.current_joint_state.position)]  # len=7


def read_joint_velocity(robot: "Robot") -> List[float]:
    return [float(dq) for dq in list(robot.current_joint_state.velocity)]  # len=7


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser("Red-ball client + Franka state overlay + dataset logger")
    # Stream source
    ap.add_argument("--server-ip", default="10.1.38.22")
    ap.add_argument("--server-port", type=int, default=5001)

    # Franka state (UDP in)
    ap.add_argument("--state-ip", default="0.0.0.0")
    ap.add_argument("--state-port", type=int, default=9091)
    ap.add_argument("--state-mode", choices=["basic", "full"], default="full",
                    help="Match the sender (--telemetry on the robot server).")

    # HSV / detection
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--h1l", type=int, default=0);   ap.add_argument("--h1u", type=int, default=10)
    ap.add_argument("--h2l", type=int, default=170); ap.add_argument("--h2u", type=int, default=180)
    ap.add_argument("--smin", type=int, default=120); ap.add_argument("--vmin", type=int, default=70)
    ap.add_argument("--open", type=int, default=3); ap.add_argument("--close", type=int, default=5)
    ap.add_argument("--minr", type=int, default=3)

    # Camera intrinsics for 3D conversion
    ap.add_argument("--fx", type=float, default=643.634, help="Focal length x (pixels)")
    ap.add_argument("--fy", type=float, default=642.773, help="Focal length y (pixels)")
    ap.add_argument("--cx", type=float, default=651.114, help="Principal point x (pixels)")
    ap.add_argument("--cy", type=float, default=367.637, help="Principal point y (pixels)")

    # Depth handling
    ap.add_argument("--depth-method", choices=["camera", "ball_size", "fixed"], default="camera",
                    help="How to obtain depth for 3D: from camera depth map, from known ball size, or fixed value")
    ap.add_argument("--fixed-depth", type=float, default=1.16, help="Fixed depth value (meters)")
    ap.add_argument("--expect-depth", action="store_true", default=True,
                    help="Expect depth data from server (keep if your server sends RGBD)")

    # Ball size (both args supported; we resolve to a diameter internally)
    ap.add_argument("--ball-diameter", type=float, default=None, help="Ball diameter in meters")
    ap.add_argument("--ball-radius", type=float, default=0.06, help="Ball radius in meters (used if diameter not provided)")

    # Optional rebroadcast of pixel coords
    ap.add_argument("--coords-ip", default=None)
    ap.add_argument("--coords-port", type=int, default=9092)

    # Dataset logging
    ap.add_argument("--log", action="store_true", help="Enable CSV logging")
    ap.add_argument("--csv", default="./data_realworld/reach/traj50.csv", help="CSV output file")

    # UI
    ap.add_argument("--show", action="store_true", help="Display window with overlays")
    args = ap.parse_args()

    # Resolve ball diameter (meters)
    ball_diameter_m = args.ball_diameter if args.ball_diameter is not None else (2.0 * args.ball_radius)

    # Connect TCP (camera)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))
    print(f"[Client] Connected to camera {args.server_ip}:{args.server_port}")

    # Optional direct robot connection (fallback only)
    robot = gripper = None
    if _HAS_FRANKY:
        try:
            robot = Robot(ROBOT_IP)
            robot.relative_dynamics_factor = 0.01  # gentle
            gripper = franky.Gripper(ROBOT_IP)
        except Exception as e:
            print(f"[Client] Franky connect failed ({e}); will rely on UDP telemetry only.")
            robot = None
            gripper = None

    # Franka state receiver (UDP)
    st_rx = FrankaStateRX(args.state_ip, args.state_port, args.state_mode)

    # Headers
    hdr_rgb_size = struct.calcsize(HEADER_RGB_FMT)
    hdr_rgbd_size = struct.calcsize(HEADER_RGBD_FMT)

    # Optional coords UDP
    coords_sock = None; coords_dst = None
    if args.coords_ip:
        coords_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        coords_dst = (args.coords_ip, args.coords_port)
        print(f"[Client] Will send coords to {coords_dst}")

    # Dataset CSV
    csv_fp, csv_w = None, None

    frames, t0 = 0, time.time()
    prev_depth = 0.0
    prev_ball = None  # (t_ns, x_w, y_w, z_w) for velocity

    frame_id = 0

    try:
        while True:
            # Receive headers / payloads
            if args.expect_depth:
                hdr = recv_all(sock, hdr_rgbd_size)
                if hdr is None:
                    print("[Client] Stream disconnected (header).")
                    break
                (rgb_length, t_cam_ns, depth_length) = struct.unpack(HEADER_RGBD_FMT, hdr)
                jpg = recv_all(sock, rgb_length)
                if jpg is None:
                    print("[Client] Stream disconnected (rgb).")
                    break
                depth_bytes = recv_all(sock, depth_length)
                if depth_bytes is None:
                    print("[Client] Stream disconnected (depth).")
                    break
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

                # NOTE: If your server sends depth as raw uint16 mm (not PNG), comment the imdecode line and use reshape.
                # depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((720, 1280))
                depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                if depth is None or depth.dtype != np.uint16:
                    # Fallback: assume raw uint16 and 720p
                    try:
                        depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((720, 1280))
                    except Exception:
                        depth = None
                        print("[Client] Depth decode failed; skipping 3D.")
            else:
                hdr = recv_all(sock, hdr_rgb_size)
                if hdr is None:
                    print("[Client] Stream disconnected (header).")
                    break
                (rgb_length, t_cam_ns) = struct.unpack(HEADER_RGB_FMT, hdr)
                jpg = recv_all(sock, rgb_length)
                if jpg is None:
                    print("[Client] Stream disconnected (rgb).")
                    break
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                depth = None

            if img is None:
                continue

            # Poll latest UDP telemetry (non-blocking)
            st = st_rx.poll()

            # Detect red ball
            hsv1 = ((args.h1l, args.smin, args.vmin), (args.h1u, 255, 255))
            hsv2 = ((args.h2l, args.smin, args.vmin), (args.h2u, 255, 255))
            t_in = time.time()
            det = detect_red_ball(img, scale=args.scale, hsv1=hsv1, hsv2=hsv2,
                                  morph_open_ksize=args.open, morph_close_ksize=args.close, min_radius_px=args.minr)
            t_out = time.time()

            cx = cy = radius = -1
            x_cam = y_cam = z_cam = 0.0
            x_w = y_w = z_w = 0.0
            vx_w = vy_w = vz_w = 0.0

            if det is not None:
                cx, cy = map(int, det["center"])
                radius = int(det["radius"])

                # Depth selection
                if args.depth_method == "ball_size":
                    depth_m = estimate_depth_from_ball_size(det["radius"], ball_diameter_m, args.fx)
                elif args.depth_method == "fixed":
                    depth_m = float(args.fixed_depth)
                elif args.depth_method == "camera" and depth is not None:
                    dmm = int(depth[cy, cx])  # mm
                    depth_m = (dmm / 1000.0) if dmm > 0 else (prev_depth if prev_depth > 0 else float(args.fixed_depth))
                else:
                    depth_m = prev_depth if prev_depth > 0 else float(args.fixed_depth)
                prev_depth = depth_m

                # Camera-frame 3D
                x_cam, y_cam, z_cam = pixel_to_camera_coords(cx, cy, depth_m, args.fx, args.fy, args.cx, args.cy)

                # Rotate camera coords (modify if your camera axes differ)
                if _HAS_SCIPY:
                    R_y = SciPyRotation.from_euler('y', 90, degrees=True).as_matrix()
                    R_x = SciPyRotation.from_euler('x', 270, degrees=True).as_matrix()
                    R_combined = R_x @ R_y
                    point_rot = R_combined @ np.array([x_cam, y_cam, z_cam], dtype=np.float64)
                    x_cam, y_cam, z_cam = float(point_rot[0]), float(point_rot[1]), float(point_rot[2])

                # A small offset in x if you need (as in your original code)
                x_cam += float(args.ball_radius)

                # World transform
                point_cam_h = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
                point_world_h = transformation @ point_cam_h
                x_w, y_w, z_w = float(point_world_h[0]), float(point_world_h[1]), float(point_world_h[2])

                # Ball velocity in world frame (simple finite difference using camera timestamps)
                if prev_ball is not None:
                    t_prev_ns, px, py, pz = prev_ball
                    dt_s = (int(t_cam_ns) - int(t_prev_ns)) / 1e9
                    if dt_s > 0:
                        vx_w = (x_w - px) / dt_s
                        vy_w = (y_w - py) / dt_s
                        vz_w = (z_w - pz) / dt_s
                prev_ball = (int(t_cam_ns), x_w, y_w, z_w)

                # Optional visualization
                if args.show:
                    cv2.circle(img, (cx, cy), radius, (0, 0, 255), 2)
                    cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.putText(img, f"red-ball {(t_out - t_in) * 1000:.1f}ms", (8, img.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, f"3D(world): ({x_w:.3f}, {y_w:.3f}, {z_w:.3f}) m",
                                (8, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Overlay state from UDP on image
            if args.show:
                img = overlay_state(img, st, args.state_mode)

            # Optional rebroadcast of pixel coords
            if coords_sock and coords_dst:
                score = 1.0 if det is not None else 0.0
                t_ms = int(time.time() * 1000) & 0xFFFFFFFF
                pkt = struct.pack(COORDS_FMT, int(cx), int(cy), int(max(0.0, min(1.0, score)) * 1000), t_ms)
                coords_sock.sendto(pkt, coords_dst)

            # -----------------------------
            # DATASET LOGGER (CSV)
            # -----------------------------
            if args.log and det is not None:
                # Init-on-first-use
                if csv_w is None:
                    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
                    csv_fp_local = open(args.csv, "w", newline="")
                    csv_w_local = csv.writer(csv_fp_local)
                    header = [
                        "t_cam_ns", "frame_id",
                        "ball_px_cx", "ball_px_cy", "ball_px_radius",
                        "ball_cam_x", "ball_cam_y", "ball_cam_z",
                        "ball_world_x", "ball_world_y", "ball_world_z",
                        "ball_world_vx", "ball_world_vy", "ball_world_vz",
                        "ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw"
                    ]
                    if args.state_mode == "full":
                        header += [
                            "ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz",
                            "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                            "gripper_ratio"
                        ]
                    else:
                        header += ["gripper_ratio"]
                    csv_w_local.writerow(header)
                    csv_fp, csv_w = csv_fp_local, csv_w_local  # capture to outer scope

                # Choose telemetry source: UDP preferred; fallback to franky reads if available
                ee = st if st is not None else {}
                if not ee and _HAS_FRANKY and robot is not None and gripper is not None:
                    try:
                        x, y, z, r, p, yaw = read_ee_pose_rpy(robot)
                        twist6 = read_cartesian_twist(robot) if args.state_mode == "full" else [float("nan")] * 6
                        q7 = read_joint_position(robot) if args.state_mode == "full" else [float("nan")] * 7
                        dq7 = read_joint_velocity(robot) if args.state_mode == "full" else [float("nan")] * 7
                        g = read_gripper_open_ratio(gripper)
                        ee = {
                            "x": x, "y": y, "z": z, "roll": r, "pitch": p, "yaw": yaw,
                            "vx": twist6[0] if args.state_mode == "full" else float("nan"),
                            "vy": twist6[1] if args.state_mode == "full" else float("nan"),
                            "vz": twist6[2] if args.state_mode == "full" else float("nan"),
                            "wx": twist6[3] if args.state_mode == "full" else float("nan"),
                            "wy": twist6[4] if args.state_mode == "full" else float("nan"),
                            "wz": twist6[5] if args.state_mode == "full" else float("nan"),
                            "q": q7, "dq": dq7, "g": g
                        }
                    except Exception:
                        ee = {}

                # Fill with NaNs if still missing
                def getf(d, k):
                    return float(d[k]) if (d is not None and k in d and d[k] is not None) else float("nan")

                row = [
                    int(t_cam_ns), int(frame_id),
                    int(cx), int(cy), float(radius),
                    float(x_cam), float(y_cam), float(z_cam),
                    float(x_w), float(y_w), float(z_w),
                    float(vx_w), float(vy_w), float(vz_w),
                    getf(ee, "x"), getf(ee, "y"), getf(ee, "z"),
                    getf(ee, "roll"), getf(ee, "pitch"), getf(ee, "yaw")
                ]
                if args.state_mode == "full":
                    q = ee.get("q", [float("nan")] * 7)
                    dq = ee.get("dq", [float("nan")] * 7)
                    row += [
                        getf(ee, "vx"), getf(ee, "vy"), getf(ee, "vz"),
                        getf(ee, "wx"), getf(ee, "wy"), getf(ee, "wz"),
                        *[float(v) for v in q[:7]],
                        *[float(v) for v in dq[:7]],
                        getf(ee, "g")
                    ]
                else:
                    row += [getf(ee, "g")]

                csv_w.writerow(row)
                if (frame_id % 60) == 0:
                    csv_fp.flush()

            # Show window
            if args.show:
                cv2.imshow("Client (HSV + Franka state)", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1
            frames += 1
            if frames % 60 == 0:
                fps = frames / (time.time() - t0 + 1e-6)
                print(f"[Client] ~{fps:.1f} FPS")

    finally:
        try:
            sock.close()
        except Exception:
            pass
        st_rx.close()
        if coords_sock:
            try:
                coords_sock.close()
            except Exception:
                pass
        if csv_fp:
            try:
                csv_fp.flush()
            except Exception:
                pass
            try:
                csv_fp.close()
            except Exception:
                pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
