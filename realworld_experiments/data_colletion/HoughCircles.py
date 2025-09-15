# pip install ultralytics opencv-python
import time
import cv2
import numpy as np
from functools import lru_cache

SPORTS_BALL_CLS = 32  # COCO id for "sports ball"
device = "cuda:0"  # or None for CPU
# ---------- Ultra-fast fallback (Hough) ----------
def _hough_ball(img, dp=1.2, min_dist_frac=0.08, param1=120, param2=26, min_r=5, max_r=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    H, W = gray.shape[:2]
    min_dist = max(8, int(min_dist_frac * max(H, W)))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    x, y, r = max(np.around(circles[0]).astype(int), key=lambda c: c[2])
    x1, y1, x2, y2 = x - r, y - r, x + r, y + r
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    return (x1, y1, x2, y2, 1.0)

# ---------- YOLO (cached, FP16 on CUDA, class-limited) ----------
@lru_cache(maxsize=1)
def _get_yolo(model_name="yolov8n.pt"):
    from ultralytics import YOLO
    m = YOLO(model_name)
    # Optional: tiny speedup by fusing layers (in-place)
    try:
        m.fuse()
    except Exception:
        pass
    # Try to push to CUDA + FP16 if available
    try:
        m.to(device)
        print("Using YOLOv8 on CUDA with FP16")
        m.model.half()
        print(m.device)
    except Exception:
        pass
    return m

def _yolo_best_ball(img, conf=0.25, imgsz=320):
    m = _get_yolo()  # cached once
    # Use direct call (preferred API), limit to class 32 and 1 detection
    results = m(img, conf=conf, imgsz=imgsz, classes=[SPORTS_BALL_CLS],
                max_det=1, verbose=False)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    b = r.boxes[0]
    x1, y1, x2, y2 = map(int, map(float, b.xyxy[0]))
    score = float(b.conf[0])
    return (x1, y1, x2, y2, score)

# ---------- Public API ----------
def detect_ball_fast(image_path, out_path=None, prefer_hough=True, imgsz=320, conf=0.25, draw=False):
    """
    Fast single-image detector aiming for >=30Hz throughput.
    Returns: (cx, cy), (x1, y1, x2, y2, score, source) or None
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    det, src = None, None

    # 1) Hough first (CPU lightning-fast). If not found, use YOLO (robust).
    if prefer_hough:
        det = _hough_ball(img)
        if det is not None:
            src = "hough"
    if det is None:
        det = _yolo_best_ball(img, conf=conf, imgsz=imgsz)
        if det is not None:
            src = "yolo"

    if det is None:
        return None

    x1, y1, x2, y2, score = det
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw or out_path:
        draw_img = img.copy()
        color = (255, 0, 0) if src == "hough" else (0, 255, 0)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(draw_img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(draw_img, f"{src} {score:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if out_path:
            cv2.imwrite(out_path, draw_img)
        if draw:
            cv2.imshow("Ball", draw_img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

    return (cx, cy), (x1, y1, x2, y2, score, src)

# ---------- Quick benchmark (throughput) ----------
def benchmark(image_path, runs=100, imgsz=320, conf=0.25, prefer_hough=True):
    # Warmup once (build CUDA graph / cudnn autotune etc.)
    detect_ball_fast(image_path, out_path="result.png", prefer_hough=prefer_hough, imgsz=imgsz, conf=conf, draw=True)
    t0 = time.perf_counter()
    ok = 0
    for _ in range(runs):
        res = detect_ball_fast(image_path, out_path=None, prefer_hough=prefer_hough, imgsz=imgsz, conf=conf, draw=False)
        ok += int(res is not None)
    dt = time.perf_counter() - t0
    fps = runs / dt
    print(f"Runs: {runs}, Hits: {ok}, Total: {dt:.3f}s, FPS: {fps:.1f}")
    return fps

if __name__ == "__main__":
    # Example: target >= 30 FPS
    # Tip: start with prefer_hough=True for round balls; if your ball isn't perfectly circular, set False.
    res = detect_ball_fast("1.png", out_path="1_annotated.png", prefer_hough=True, imgsz=320, conf=0.25, draw=False)
    print("Result:", res)
    benchmark("1.png", runs=120, imgsz=320, conf=0.25, prefer_hough=True)
