#!/usr/bin/env python3
"""
Franka Action/State Server (UDP telemetry)
- Sends Franka state (basic or full) to a destination (e.g., your PC) via UDP.
- Optionally receives actions (disabled by default in this script).

Packets (little-endian):
- basic "<7d":  [x, y, z, roll, pitch, yaw, gripper_ratio]
- full  "<27d": [x, y, z, roll, pitch, yaw,
                 vx, vy, vz, wx, wy, wz,
                 q1..q7, dq1..dq7,
                 gripper_ratio]
"""
import argparse
import math
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# Optional: franky / libfranka
try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None

try:
    from franky import Robot, Gripper
    import franky
except Exception as e:
    Robot = None
    Gripper = None
    franky = None

def rpy_from_quat(qx, qy, qz, qw):
    if R is None:
        # Fallback approximate conversion
        ysqr = qy * qy
        t0 = +2.0 * (qw * qx + qy * qz)
        t1 = +1.0 - 2.0 * (qx * qx + ysqr)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (qw * qy - qz * qx)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (qw * qz + qx * qy)
        t4 = +1.0 - 2.0 * (ysqr + qz * qz)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw
    else:
        return R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)

def now():
    return time.time()

def now_ns():
    return time.time_ns()

@dataclass
class TelemetryConfig:
    mode: str = "full"   # "basic" or "full"
    rate_hz: float = 100.0

STATE_FMT_BASIC = "<7d"
STATE_FMT_FULL  = "<27d"

def read_robot_state(robot: "Robot", gripper: "Gripper"):
    """
    Returns a dict with pose (x,y,z,r,p,yaw), twist (vx..wz), joints q1..q7, dq1..dq7, and gripper_ratio.
    Assumes Robot provides current pose (translation + quaternion), velocity, and joint states.
    """
    # The franky API shape may vary; adapt here to your actual attributes
    # Placeholder interface:
    # pose: {'xyz': (x,y,z), 'quat': (qx,qy,qz,qw)}
    # twist: {'v': (vx,vy,vz), 'w': (wx,wy,wz)}
    # joints: {'q': (q1..q7), 'dq': (dq1..dq7)}
    # gripper: open ratio 0..1
    try:
        pose = robot.current_pose()        # expect returns (x,y,z,qx,qy,qz,qw) or dict
    except Exception:
        # Some APIs use .current_pose instead of function
        pose = robot.current_pose

    if isinstance(pose, tuple) and len(pose) == 7:
        x, y, z, qx, qy, qz, qw = pose
    elif isinstance(pose, dict):
        (x, y, z) = pose.get('xyz', (0,0,0))
        (qx, qy, qz, qw) = pose.get('quat', (0,0,0,1))
    else:
        # Fallback: zeros
        x=y=z=0.0
        qx=qy=qz=0.0
        qw=1.0

    roll, pitch, yaw = rpy_from_quat(qx, qy, qz, qw)

    try:
        twist = robot.current_twist()
    except Exception:
        twist = getattr(robot, "current_twist", {'v': (0,0,0), 'w': (0,0,0)})
    if isinstance(twist, dict):
        vx, vy, vz = twist.get('v', (0,0,0))
        wx, wy, wz = twist.get('w', (0,0,0))
    elif isinstance(twist, tuple) and len(twist) == 6:
        vx, vy, vz, wx, wy, wz = twist
    else:
        vx=vy=vz=wx=wy=wz=0.0

    try:
        joints = robot.current_joints()
    except Exception:
        joints = getattr(robot, "current_joints", {'q': (0,)*7, 'dq': (0,)*7})

    if isinstance(joints, dict):
        q = joints.get('q', (0,)*7)
        dq = joints.get('dq', (0,)*7)
    elif isinstance(joints, tuple) and len(joints) >= 14:
        q = joints[:7]
        dq = joints[7:14]
    else:
        q = (0.0,)*7
        dq = (0.0,)*7

    try:
        # 0..1 opening ratio
        g = gripper.current_opening_ratio()
    except Exception:
        g = 0.0

    return dict(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                vx=vx, vy=vy, vz=vz, wx=wx, wy=wy, wz=wz,
                q=list(q), dq=list(dq), g=g)

def pack_basic(st: dict) -> bytes:
    return struct.pack(STATE_FMT_BASIC,
        float(st['x']), float(st['y']), float(st['z']),
        float(st['roll']), float(st['pitch']), float(st['yaw']),
        float(st['g'])
    )

def pack_full(st: dict) -> bytes:
    data = [
        float(st['x']), float(st['y']), float(st['z']),
        float(st['roll']), float(st['pitch']), float(st['yaw']),
        float(st['vx']), float(st['vy']), float(st['vz']),
        float(st['wx']), float(st['wy']), float(st['wz'])
    ]
    data.extend([float(v) for v in st['q'][:7]])
    data.extend([float(v) for v in st['dq'][:7]])
    data.append(float(st['g']))
    return struct.pack(STATE_FMT_FULL, *data)

def telemetry_loop(robot_ip: str, state_dst: Tuple[str,int], cfg: TelemetryConfig, stop_evt: threading.Event):
    if Robot is None:
        print("[Franka] franky not available. Exiting.")
        return
    print(f"[Franka] Connecting to robot at {robot_ip} ...")
    robot = Robot(robot_ip)
    gripper = Gripper(robot_ip)
    print("[Franka] Connected. Starting telemetry:", cfg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    period = 1.0 / max(1e-3, cfg.rate_hz)
    last_log = 0.0

    while not stop_evt.is_set():
        t0 = time.time()
        st = read_robot_state(robot, gripper)
        if cfg.mode == "basic":
            pkt = pack_basic(st)
        else:
            pkt = pack_full(st)
        sock.sendto(pkt, state_dst)

        if (time.time() - last_log) > 1.0:
            last_log = time.time()
            print(f"[Franka] Sent {cfg.mode} pose=({st['x']:.3f},{st['y']:.3f},{st['z']:.3f}) "
                  f"rpy=({st['roll']:.2f},{st['pitch']:.2f},{st['yaw']:.2f}) g={st['g']:.2f}")
        # sleep to maintain rate
        dt = time.time() - t0
        to_sleep = period - dt
        if to_sleep > 0:
            time.sleep(to_sleep)

    sock.close()
    print("[Franka] Telemetry loop stopped.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", type=str, required=True, help="Franka robot IP (FCI)")
    ap.add_argument("--state-dst-ip", type=str, default="127.0.0.1", help="Where to send telemetry (your PC IP)")
    ap.add_argument("--state-dst-port", type=int, default=9091, help="UDP port for telemetry")
    ap.add_argument("--telemetry", choices=["basic", "full"], default="full")
    ap.add_argument("--state_rate", type=float, default=30.0, help="Telemetry rate Hz")
    args = ap.parse_args()

    cfg = TelemetryConfig(mode=args.telemetry, rate_hz=args.state_rate)
    stop_evt = threading.Event()
    try:
        telemetry_loop(args.robot_ip, (args.state_dst_ip, args.state_dst_port), cfg, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()

if __name__ == "__main__":
    main()
