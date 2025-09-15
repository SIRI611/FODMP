#!/usr/bin/env python3
import socket, struct, time, argparse, math

FMT = "<7d"  # dx,dy,dz,droll,dpitch,dyaw,grip

def make_action(t):
    """Demo: small sine on x. Replace with your policy."""
    dx = 0.01 * math.sin(t)   # meters (relative)
    dy = 0.00
    dz = 0.00
    droll = dpitch = dyaw = 0.0  # radians (relative)
    grip = 0.0  # 0=open, 1=close
    return (dx, dy, dz, droll, dpitch, dyaw, grip)

def main():
    ap = argparse.ArgumentParser("YOLO PC -> send Franka actions")
    ap.add_argument("--dest-ip", default="10.1.38.22", help="Camera PC IP")
    ap.add_argument("--dest-port", type=int, default=9090)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--demo", action="store_true", help="run demo action generator")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_ip, args.dest_port)
    period = 1.0 / max(args.hz, 1.0)

    print(f"[sender] sending to {dest} at {args.hz:.1f} Hz. Ctrl+C to stop.")
    t0 = time.time()
    try:
        while True:
            if args.demo:
                act = make_action(time.time() - t0)
            else:
                # TODO: plug your own action here, e.g. from ball coords/state
                act = make_action(time.time() - t0)

            sock.sendto(struct.pack(FMT, *act), dest)
            time.sleep(period)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
