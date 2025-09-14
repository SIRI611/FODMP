#!/usr/bin/env python3
import socket, struct, time, argparse, math
from typing import Tuple
import numpy as np

from scipy.spatial.transform import Rotation  # pip install scipy
from franky import *
import franky

# ---------- Network formats ----------
ACT_FMT   = "<7d"   # dx,dy,dz,droll,dpitch,dyaw,grip     (YOLO PC -> Camera PC)
STATE_FMT = "<7d"   # x,y,z,roll,pitch,yaw,gripper        (Camera PC -> YOLO PC)
ACT_BYTES = struct.calcsize(ACT_FMT)

# ---------- Safety & scaling ----------
# Clip per-command relative steps
MAX_DPOS   = 0.02      # m per command
MAX_DANG   = 0.10      # rad per command
# Scale incoming deltas (so sender can be in small normalized units if you prefer)
LIN_SCALE  = 1.0       # multiply dx,dy,dz  (meters)
ANG_SCALE  = 1.0       # multiply droll,dpitch,dyaw (radians)
# Motion pacing
COMMAND_DELAY = 0.03   # s small settle between move() calls

# ---------- Helpers ----------
def clamp(v, lo, hi): return max(lo, min(hi, v))

def euler_xyz_to_quat_wxyz(rx, ry, rz):
    # from_euler returns [x,y,z,w]
    q = Rotation.from_euler("xyz", [rx, ry, rz]).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]  # -> [w,x,y,z]

def quat_to_rpy_wxyz(w, x, y, z) -> Tuple[float,float,float]:
    # Yaw-Pitch-Roll (ZYX intrinsic)
    t0 = +2.0*(w*x + y*z); t1 = +1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(t0, t1)
    t2 = +2.0*(w*y - z*x); t2 = 1.0 if t2>1.0 else (-1.0 if t2<-1.0 else t2)
    pitch = math.asin(t2)
    t3 = +2.0*(w*z + x*y); t4 = +1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def read_state(robot: Robot, grip: franky.Gripper):
    cs = robot.current_cartesian_state
    pose = cs.pose.end_effector_pose
    x,y,z = pose.translation
    w,xq,yq,zq = pose.quaternion
    r,p,yw = quat_to_rpy_wxyz(w,xq,yq,zq)
    try:
        width = float(grip.width)  # m
        # Normalize (0=open, 1=close) if you prefer; here we send raw width in meters:
        g = width
    except Exception:
        g = float("nan")
    return (x,y,z,r,p,yw,g)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser("Camera PC Franka control server")
    ap.add_argument("--robot-ip", default="10.1.38.23")
    ap.add_argument("--listen-ip", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=9090, help="actions in")
    ap.add_argument("--state-dest-ip", default="10.1.38.24", help="YOLO PC (optional state out)")
    ap.add_argument("--state-dest-port", type=int, default=9091)
    ap.add_argument("--state-hz", type=float, default=50.0)
    ap.add_argument("--no-state", action="store_true", help="disable state UDP publisher")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # --- Connect robot ---
    print(f"[franka] connecting to {args.robot_ip} … (enable FCI & whitelist this PC in Desk)")
    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = 0.01
    grip = franky.Gripper(args.robot_ip)

    # Affirm current pose (absolute no-op; helps ensure ready state)
    try:
        cs = robot.current_cartesian_state
        pose = cs.pose.end_effector_pose
        robot.move(CartesianMotion(Affine(list(pose.translation), list(pose.quaternion)),
                                   ReferenceType.Absolute))
        print("[franka] ready.")
    except Exception as e:
        print("[franka] warning on init move:", e)

    # --- Sockets ---
    act_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    act_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    act_sock.bind((args.listen_ip, args.listen_port))
    act_sock.settimeout(0.05)

    state_sock = None; state_dest = None; state_period = None; last_state = 0.0
    if not args.no_state:
        state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_dest = (args.state_dest_ip, args.state_dest_port)
        state_period = 1.0 / max(args.state_hz, 1.0)

    print(f"[server] actions @ {args.listen_ip}:{args.listen_port}  "
          f"{'(state-> ' + str(state_dest) + f' @ {args.state_hz:.0f}Hz)' if state_dest else '(no state)'}")

    last_grip_mode = None
    GRIPPER_COOLDOWN = 0.4
    last_grip_t = 0.0

    try:
        while True:
            # 1) Receive latest action (non-blocking-ish)
            try:
                data, addr = act_sock.recvfrom(1024)
            except socket.timeout:
                data = None

            if data and len(data) >= ACT_BYTES:
                dx,dy,dz, droll,dpitch,dyaw, grip_cmd = struct.unpack(ACT_FMT, data[:ACT_BYTES])

                # Scale & clamp for safety
                dx = clamp(dx * LIN_SCALE, -MAX_DPOS, MAX_DPOS)
                dy = clamp(dy * LIN_SCALE, -MAX_DPOS, MAX_DPOS)
                dz = clamp(dz * LIN_SCALE, -MAX_DPOS, MAX_DPOS)
                droll  = clamp(droll  * ANG_SCALE, -MAX_DANG, MAX_DANG)
                dpitch = clamp(dpitch * ANG_SCALE, -MAX_DANG, MAX_DANG)
                dyaw   = clamp(dyaw   * ANG_SCALE, -MAX_DANG, MAX_DANG)

                # Relative orientation as quaternion [w,x,y,z]
                dq_wxyz = euler_xyz_to_quat_wxyz(droll, dpitch, dyaw)

                # Execute relative Cartesian motion
                try:
                    robot.move(CartesianMotion(Affine([dx,dy,dz], dq_wxyz), ReferenceType.Relative))
                    if not args.quiet:
                        print(f"[move] d=({dx:+.3f},{dy:+.3f},{dz:+.3f})  "
                              f"drpy=({droll:+.2f},{dpitch:+.2f},{dyaw:+.2f})")
                    time.sleep(COMMAND_DELAY)
                except Exception as e:
                    print("[move] error:", e)

                # Gripper: only act on clear open/close commands
                try:
                    mode = "open" if grip_cmd <= 0.0 else ("close" if grip_cmd >= 1.0 else None)
                    now = time.time()
                    if mode and (mode != last_grip_mode) and (now - last_grip_t >= GRIPPER_COOLDOWN):
                        if mode == "close":
                            # example: close with force; adjust to your jaws & object
                            grip.grasp(0.0, 0.05, 20.0, epsilon_outer=1.0)
                            print("[gripper] close")
                        else:
                            grip.open(0.05)
                            print("[gripper] open")
                        last_grip_mode = mode; last_grip_t = now
                except Exception as e:
                    print("[gripper] error:", e)

            # 2) Periodic state publisher
            if state_sock and (time.time() - last_state >= state_period):
                try:
                    x,y,z,r,p,yw,g = read_state(robot, grip)
                    pkt = struct.pack(STATE_FMT, x,y,z,r,p,yw,g)
                    state_sock.sendto(pkt, state_dest)
                except Exception as e:
                    print("[state] send error:", e)
                last_state = time.time()

    except KeyboardInterrupt:
        print("\n[server] Ctrl+C, exiting.")
    finally:
        try: act_sock.close()
        except: pass
        if state_sock:
            try: state_sock.close()
            except: pass

if __name__ == "__main__":
    main()
