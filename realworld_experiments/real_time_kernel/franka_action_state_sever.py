#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franka (franky) incremental Cartesian position control with UDP I/O.

- Listens for action packets from a YOLO/control PC: <dx, dy, dz, droll, dpitch, dyaw, grip_cmd> (all float64)
- Executes small relative Cartesian moves using franky.CartesianMotion(..., ReferenceType.Relative)
- Publishes robot telemetry to the YOLO/control PC via UDP at a fixed rate
- Telemetry formats:
    basic (default): 7 doubles  -> [x, y, z, roll, pitch, yaw, gripper_ratio]
    full:            27 doubles -> [x y z r p y] + [vx vy vz wx wy wz] + [q1..q7] + [dq1..dq7] + [gripper_ratio]
- Quaternion delta is computed via SciPy if available; otherwise falls back to a numerically-stable pure-Python path.

Tested API surfaces (adjust if your franky version differs):
    - Robot.current_cartesian_state.pose.end_effector_pose.translation
    - Robot.current_cartesian_state.pose.end_effector_pose.quaternion  # [w, x, y, z]
    - Robot.current_cartesian_state.velocity.end_effector_twist        # [vx,vy,vz,wx,wy,wz]
    - Robot.current_joint_state.position                                # 7
    - Robot.current_joint_state.velocity                                # 7
    - Gripper.open(width_speed_mps) / Gripper.move(width_m, speed) / Gripper.grasp(width_m, speed, force)

Author: you :)
"""

import argparse
import math
import socket
import struct
import time
from typing import Tuple, List

import numpy as np

# --- SciPy is optional: we use it if present; otherwise fall back to pure math
try:
    from scipy.spatial.transform import Rotation as SciPyRotation  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from franky import *
import franky


# =========================
# User-configurable defaults
# =========================
ROBOT_IP = "10.1.38.23"

# YOLO/control PC where we publish telemetry
PC1_IP = "10.1.38.24"
STATE_DST = (PC1_IP, 9091)

# Action receive bind (this PC)
DEFAULT_LISTEN_IP = "0.0.0.0"
DEFAULT_LISTEN_PORT = 9090

# Telemetry rate
STATE_HZ = 30.0
STATE_PERIOD = 1.0 / STATE_HZ

# Safety scaling for incoming deltas
DELTA_SCALE_POS = 0.05   # scale dx,dy,dz
DELTA_SCALE_ROT = 0.05   # scale droll,dpitch,dyaw

# Small settle so we don't hammer the control loop
MOVE_SLEEP = 0.03  # seconds (~33 Hz)

# Gripper
GRIP_SPEED = 0.02  # m/s (adjust to your hardware)
GRIP_FORCE = 20.0  # N (adjust to your hardware)
GRIPPER_COOLDOWN = 0.5  # seconds between mode toggles
GRIPPER_MAX_OPEN = 0.08  # meters
GRIPPER_MIN_OPEN = 0.00  # meters

# Optional initial absolute EE pose (if you want to force an initial move)
Q_INIT_XYZ = np.array([0.31497584, -0.00210619, 0.44722789])  # meters
QUAT_INIT_WXYZ = np.array([0.99416457, 0.0590707, 0.08990171, -0.00807068])  # [w,x,y,z]


# =========================
# Utilities
# =========================
def make_socket(bind_ip: str, bind_port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip, bind_port))
    sock.settimeout(0.1)  # semi-nonblocking
    return sock


def quat_to_rpy_wxyz(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert quaternion (w, x, y, z) to ZYX (yaw-pitch-roll) Euler, return (roll, pitch, yaw)."""
    # Based on standard conversions; numerically stable clamping on asin input
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
    """Pure-Python delta quaternion from small XYZ intrinsic Euler angles -> [w,x,y,z]."""
    hr, hp, hy = rx * 0.5, ry * 0.5, rz * 0.5
    cr, sr = math.cos(hr), math.sin(hr)
    cp, sp = math.cos(hp), math.sin(hp)
    cy, sy = math.cos(hy), math.sin(hy)
    # XYZ intrinsic composition quaternion
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return [w, x, y, z]


def delta_quat_wxyz_from_euler_xyz(droll: float, dpitch: float, dyaw: float) -> List[float]:
    """Prefer SciPy if available; otherwise use a stable pure-Python formula."""
    if _HAS_SCIPY:
        qxyzw = SciPyRotation.from_euler("xyz", [droll, dpitch, dyaw]).as_quat()  # [x, y, z, w]
        return [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]  # -> [w, x, y, z]
    return small_euler_xyz_to_quat_wxyz(droll, dpitch, dyaw)


# =========================
# Robot/Gripper helpers
# =========================
def read_ee_pose_rpy(robot: Robot) -> Tuple[float, float, float, float, float, float]:
    """
    Returns the end-effector pose of the robot in ZYX (yaw-pitch-roll) Euler angles.

    :param robot: the Franka robot object
    :return: (x, y, z, roll, pitch, yaw) as floats
    """
    cart = robot.current_cartesian_state
    pose = cart.pose.end_effector_pose
    x, y, z = pose.translation  # meters
    w, xq, yq, zq = pose.quaternion  # franky provides (w, x, y, z)
    r, p, yaw = quat_to_rpy_wxyz(w, xq, yq, zq)  # ZYX Euler angles
    return float(x), float(y), float(z), float(r), float(p), float(yaw)


def read_gripper_open_ratio(gripper: franky.Gripper) -> float:
    """0.0 = fully open, 1.0 = fully closed (based on jaw width)."""
    try:
        width = float(gripper.width)
        width = max(GRIPPER_MIN_OPEN, min(GRIPPER_MAX_OPEN, width))
        closed_ratio = 1.0 - (width - GRIPPER_MIN_OPEN) / (GRIPPER_MAX_OPEN - GRIPPER_MIN_OPEN)
        return float(closed_ratio)
    except Exception:
        return 0.0


def read_cartesian_twist(robot: Robot) -> list[float]:
    """
    Return [vx, vy, vz, wx, wy, wz] from franky cartesian velocity.
    Works across multiple franky versions by probing common field names.
    """
    vel_obj = robot.current_cartesian_state.velocity
    # Some franky versions expose .end_effector_twist, others just the twist directly
    tw = getattr(vel_obj, "end_effector_twist", None)
    if tw is None:
        tw = getattr(vel_obj, "twist", None)
    if tw is None:
        tw = vel_obj  # last-resort: maybe vel_obj itself is the Twist

    # Try common (linear, angular) pairs
    for lin_name, ang_name in [
        ("linear", "angular"),
        ("translational", "rotational"),
        ("v", "w"),
        ("linear_velocity", "angular_velocity"),
    ]:
        lin = getattr(tw, lin_name, None)
        ang = getattr(tw, ang_name, None)
        if lin is not None and ang is not None:
            lin = list(lin) if hasattr(lin, "__iter__") else [lin.x, lin.y, lin.z]
            ang = list(ang) if hasattr(ang, "__iter__") else [ang.x, ang.y, ang.z]
            return [float(l) for l in lin[:3]] + [float(a) for a in ang[:3]]

    # Try direct scalar components
    comp_names = ("vx", "vy", "vz", "wx", "wy", "wz")
    if all(hasattr(tw, n) for n in comp_names):
        return [float(getattr(tw, n)) for n in comp_names]

    # If it happens to be a 6-length sequence (rare)
    try:
        seq = list(tw)
        if len(seq) == 6:
            return [float(v) for v in seq]
    except TypeError:
        pass

    # Couldn’t recognize schema
    raise TypeError(f"Unsupported Twist schema: has {dir(tw)}")



def read_joint_position(robot: Robot) -> List[float]:
    return [float(q) for q in list(robot.current_joint_state.position)]  # len=7


def read_joint_velocity(robot: Robot) -> List[float]:
    return [float(dq) for dq in list(robot.current_joint_state.velocity)]  # len=7


# =========================
# Telemetry senders
# =========================
def send_state_basic(state_sock: socket.socket, robot: Robot, gripper: franky.Gripper, dst) -> None:
    """
    7 doubles: [x, y, z, roll, pitch, yaw, gripper_ratio]
    """
    STATE_FMT = "<7d"
    x, y, z, r, p, yaw = read_ee_pose_rpy(robot)
    print(f"[State/basic] (x,y,z)=({x:.3f},{y:.3f},{z:.3f}) "
          f"(r,p,y)=({r:.2f},{p:.2f},{yaw:.2f}) g={g:.3f}")
    g = read_gripper_open_ratio(gripper)
    pkt = struct.pack(STATE_FMT, x, y, z, r, p, yaw, g)
    state_sock.sendto(pkt, dst)


def send_state_full(state_sock: socket.socket, robot: Robot, gripper: franky.Gripper, dst) -> None:
    """
    27 doubles:
      [x y z r p y] + [vx vy vz wx wy wz] + [q1..q7] + [dq1..dq7] + [g]
    """
    STATE_FMT = "<27d"
    x, y, z, r, p, yaw = read_ee_pose_rpy(robot)      # 6
    twist6 = read_cartesian_twist(robot)              # 6
    q7 = read_joint_position(robot)                   # 7
    dq7 = read_joint_velocity(robot)                  # 7
    g = read_gripper_open_ratio(gripper)              # 1
    flat = [x, y, z, r, p, yaw] + twist6 + q7 + dq7 + [g]
    pkt = struct.pack(STATE_FMT, *flat)
    state_sock.sendto(pkt, dst)


# =========================
# Main loop
# =========================
def main():
    parser = argparse.ArgumentParser(description="Franka action receiver and telemetry publisher (franky).")
    parser.add_argument("--ip", type=str, default=DEFAULT_LISTEN_IP, help="Listen IP for actions (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT, help="Listen port for actions (default: 9090)")
    parser.add_argument("--quiet", action="store_true", help="Only print errors")
    parser.add_argument("--no-init-move", type=bool, help="Skip initial absolute move", default=True)
    parser.add_argument("--telemetry", type=str, choices=["basic", "full"], default="basic",
                        help="Telemetry packet: 'basic'(7d) or 'full'(27d)")
    parser.add_argument("--receive-action", type=bool, default=False,
                        help="Receive action commands (default: False)")
    args = parser.parse_args()

    # Connect robot & gripper
    robot = Robot(ROBOT_IP)
    robot.relative_dynamics_factor = 0.01  # gentle
    gripper = franky.Gripper(ROBOT_IP)

    # Optional initial absolute pose move
    if not args.no_init_move:
        try:
            initial_motion = CartesianMotion(Affine(list(Q_INIT_XYZ), list(QUAT_INIT_WXYZ)), ReferenceType.Absolute)
            robot.move(initial_motion)
            if not args.quiet:
                print("[init] Moved to provided initial pose.")
        except Exception as e:
            print("[init] Initial pose move skipped/failed:", e)
        # Affirm current pose once more (no-op absolute)
        try:
            cart = robot.current_cartesian_state
            pose = cart.pose.end_effector_pose
            xyz = list(pose.translation)
            quat_wxyz = list(pose.quaternion)
            robot.move(CartesianMotion(Affine(xyz, quat_wxyz), ReferenceType.Absolute))
            if not args.quiet:
                print("[init] Affirmed current pose.")
        except Exception as e:
            print("[init] Skipping affirmation move:", e)

    # Sockets
    action_sock = make_socket(args.ip, args.port)
    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    FMT_ACT = "<7d"  # dx, dy, dz, droll, dpitch, dyaw, grip_cmd
    ACT_BYTES = struct.calcsize(FMT_ACT)

    last_state_send = 0.0
    last_gripper_mode = None
    last_grip_time = 0.0

    if not args.quiet:
        print(f"[recv] Listening actions on {args.ip}:{args.port} (format <7d>), "
              f"publishing telemetry to {STATE_DST[0]}:{STATE_DST[1]} ({args.telemetry})")
        print(f"[info] SciPy rotation: {'ON' if _HAS_SCIPY else 'OFF (pure-Python)'}")

    try:
        while True:
            # ---- 1) Receive action (non-blocking) ----
            data = None
            try:
                data, addr = action_sock.recvfrom(1024)
            except socket.timeout:
                pass

            if data and args.receive_action:
                if len(data) >= ACT_BYTES:
                    dx, dy, dz, droll, dpitch, dyaw, grip_cmd = struct.unpack(FMT_ACT, data[:ACT_BYTES])

                    # Safety scaling
                    dx *= DELTA_SCALE_POS
                    dy *= DELTA_SCALE_POS
                    dz *= DELTA_SCALE_POS
                    droll *= DELTA_SCALE_ROT
                    dpitch *= DELTA_SCALE_ROT
                    dyaw *= DELTA_SCALE_ROT

                    # Delta quaternion
                    dquat_wxyz = delta_quat_wxyz_from_euler_xyz(droll, dpitch, dyaw)

                    # Perform small relative Cartesian move
                    try:
                        robot.move(CartesianMotion(Affine([dx, dy, dz], dquat_wxyz), ReferenceType.Relative))
                        if not args.quiet:
                            print(f"[move] dX={dx:+.4f} dY={dy:+.4f} dZ={dz:+.4f} | "
                                  f"dR={droll:+.4f} dP={dpitch:+.4f} dY={dyaw:+.4f}")
                        time.sleep(MOVE_SLEEP)
                    except Exception as e:
                        print("[move] error:", e)

                    # Gripper mode: 0=open, 1=close, ignore in-between
                    try:
                        mode = "open" if grip_cmd <= 0.0 else ("close" if grip_cmd >= 1.0 else last_gripper_mode)
                        now = time.time()
                        if mode != last_gripper_mode and mode is not None and (now - last_grip_time) >= GRIPPER_COOLDOWN:
                            if mode == "close":
                                # Close (simple & robust)
                                gripper.move(GRIPPER_MIN_OPEN, GRIP_SPEED)
                                if not args.quiet:
                                    print("[gripper] close")
                            else:
                                # Open fully
                                gripper.move(GRIPPER_MAX_OPEN, GRIP_SPEED)
                                if not args.quiet:
                                    print("[gripper] open")
                            last_gripper_mode = mode
                            last_grip_time = now
                    except Exception as e:
                        print("[gripper] error:", e)
                else:
                    if not args.quiet:
                        print(f"[warn] short action packet {len(data)}B from {addr}, expected {ACT_BYTES}B")

            # ---- 2) Publish telemetry at fixed rate ----
            now = time.time()
            if now - last_state_send >= STATE_PERIOD:
                try:
                    if args.telemetry == "basic":
                        send_state_basic(state_sock, robot, gripper, STATE_DST)
                    else:
                        send_state_full(state_sock, robot, gripper, STATE_DST)
                except Exception as e:
                    print("[state] send error:", e)
                last_state_send = now

    except KeyboardInterrupt:
        print("\n[recv] Ctrl+C received, exiting...")
    finally:
        try:
            action_sock.close()
        except Exception:
            pass
        try:
            state_sock.close()
        except Exception:
            pass
        print("[recv] Closed.")


if __name__ == "__main__":
    main()
