# yolo_ball_detect.py
import argparse
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO
import time
SPORTS_BALL_CLS = 32 
# def _pick_best_ball(boxes, confs) -> Optional[int]:
#     """Return index of the highest-confidence detection among provided boxes, or None."""
#     if len(confs) == 0:
#         return None
#     return int(np.argmax(confs))


# def detect_ball_in_frame(model: YOLO, frame: np.ndarray) -> Optional[Dict[str, Any]]:
#     """
#     Detect a football/soccer ball in a BGR frame and return its coordinates.

#     Returns a dict with:
#       - center: (cx_px, cy_px) integer pixel coordinates
#       - normalized_center: (cx, cy) normalized to [0,1]
#       - bbox: [x1, y1, x2, y2] integers
#       - confidence: float
#       - class_name: str
#     Or None if no ball is found.
#     """
#     # Run YOLO
#     results = model(frame, verbose=False)[0]  # single image result
#     h, w = results.orig_shape  # (height, width)

#     names = results.names  # class-id -> name dict
#     # Find the COCO "sports ball" class id (usually 32). Fallback to 32 if present.
#     ball_cls_id = next((cid for cid, n in names.items() if n.lower() == "sports ball"), None)
#     if ball_cls_id is None:
#         # If the loaded model isn't COCO or lacks sports ball, nothing to do
#         return None

#     boxes_xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.zeros((0, 4))
#     boxes_cls = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.zeros((0,), dtype=int)
#     boxes_conf = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.zeros((0,))

#     # Filter only sports ball detections
#     idxs: List[int] = [i for i, c in enumerate(boxes_cls) if c == ball_cls_id]
#     if not idxs:
#         return None

#     filt_boxes = boxes_xyxy[idxs]
#     filt_confs = boxes_conf[idxs]

#     best_i = _pick_best_ball(filt_boxes, filt_confs)
#     if best_i is None:
#         return None

#     x1, y1, x2, y2 = filt_boxes[best_i].tolist()
#     conf = float(filt_confs[best_i])

#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0

#     result = {
#         "center": (int(round(cx)), int(round(cy))),
#         "normalized_center": (cx / w, cy / h),
#         "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
#         "confidence": conf,
#         "class_name": names[ball_cls_id],
#     }
#     return result


# def main():
#     parser = argparse.ArgumentParser(description="YOLO football/soccer ball detector")
#     parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLO weights (COCO-trained).")
#     parser.add_argument("--source", default="0",
#                         help="Image/video path or camera index (e.g. '0' for webcam).")
#     parser.add_argument("--device", default=None, help="Device (e.g. 'cpu', 'cuda:0').")
#     parser.add_argument("--show", action="store_true", help="Visualize detections.")
#     args = parser.parse_args()

#     model = YOLO(args.weights)

#     # Determine if source is camera index
#     is_cam = args.source.isdigit() and len(args.source) < 4
#     if is_cam:
#         cap = cv2.VideoCapture(int(args.source))
#         if not cap.isOpened():
#             raise RuntimeError("Cannot open camera.")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             res = model(frame, device=args.device, verbose=False)[0]
#             # Reuse the function for consistency
#             det = detect_ball_in_frame(model, frame)
#             if det is not None:
#                 (cx, cy) = det["center"]
#                 (x1, y1, x2, y2) = det["bbox"]
#                 print(f"Ball @ center(px)={det['center']} normalized={det['normalized_center']}"
#                       f" conf={det['confidence']:.3f}")

#                 if args.show:
#                     cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"{det['class_name']} {det['confidence']:.2f}",
#                                 (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             if args.show:
#                 cv2.imshow("Ball Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#         cap.release()
#         cv2.destroyAllWindows()

#     else:
#         # Image or video path
#         if any(args.source.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv"]):
#             cap = cv2.VideoCapture(args.source)
#             if not cap.isOpened():
#                 raise RuntimeError(f"Cannot open video: {args.source}")
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 det = detect_ball_in_frame(model, frame)
#                 if det is not None:
#                     print(f"Ball @ center(px)={det['center']} normalized={det['normalized_center']}"
#                           f" conf={det['confidence']:.3f}")
#                 if args.show:
#                     if det is not None:
#                         cx, cy = det["center"]
#                         x1, y1, x2, y2 = det["bbox"]
#                         cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(frame, f"{det['class_name']} {det['confidence']:.2f}",
#                                     (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     cv2.imshow("Ball Detection", frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#             cap.release()
#             cv2.destroyAllWindows()
#         else:
#             # Single image
#             img = cv2.imread(args.source)
#             if img is None:
#                 raise RuntimeError(f"Cannot read image: {args.source}")
#             det = detect_ball_in_frame(model, img)
#             if det is None:
#                 print("No ball detected.")
#             else:
#                 print(det)
#             if args.show:
#                 if det is not None:
#                     cx, cy = det["center"]
#                     x1, y1, x2, y2 = det["bbox"]
#                     cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# pip install ultralytics opencv-python
import cv2
from ultralytics import YOLO
def get_model(model_name="yolov4n.pt", device=None, use_half=True, fuse=True):
    """
    Load the model once and cache it. 
    - device: None (auto), 'cpu', 'cuda:0', etc.
    - use_half: FP16 on CUDA for speed
    - fuse: fuse Conv+Bn layers for a tiny speedup
    """
    m = YOLO(model_name)
    # move to device (ultralytics handles device in call(), but pushing once helps)
    # Note: safe if device is None; Ultralytics will choose.
    if fuse:
        m.fuse()  # in-place, does not return
    try:
        if device is not None:
            m.to(device)
        # half precision if CUDA
        if use_half and (device is not None and "cuda:0" in device):
            m.model.half()
    except Exception:
        # fall back silently if half/device not supported
        pass
    return m

def find_ball_yolo(image_path, model_name="yolov10n.pt", conf=0.25, show=True):
    """
    Returns (x, y) center coordinate of the highest-confidence sports ball detection.
    Also returns (x1, y1, x2, y2, score) for the chosen box.
    If show=True, displays the image with a square (bounding box) around the ball.
    """
    model = get_model(model_name=model_name, device="cuda:0", use_half=True, fuse=True)
    results = model.predict(image_path, conf=conf, verbose=False)

    r = results[0]
    # if r.boxes is None or len(r.boxes) == 0:
    #     return None
    boxes = []
    for b in r.boxes:
        cls_id = int(b.cls[0])
        score = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        # COCO: sports ball = 32
        if cls_id == 32:
            boxes.append((score, x1, y1, x2, y2))

    # if not boxes:
    #     return None

    score, x1, y1, x2, y2 = max(boxes, key=lambda t: t[0])
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    if show:
        img = cv2.imread(image_path)
        # Draw bounding box (square)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Draw center point
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imshow("Detected Ball", img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return (cx, cy), (int(x1), int(y1), int(x2), int(y2), score)

def find_ball_yolo_fast(
    image_path,
    model_name="yolov8n.pt",
    conf=0.25,
    imgsz=320,
    device=None,
    show=False
):
    """
    Fast single-image detector.
    Returns: (cx, cy), (x1, y1, x2, y2, score)  or  None
    """
    # 1) Read image once (pass array for zero-copy)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    model = get_model(model_name, device=device, use_half=True, fuse=True)

    # 3) Inference: restrict to class 32 and a small imgsz for speed
    # Prefer direct call over .predict()
    results = model(img, conf=conf, imgsz=imgsz, classes=[SPORTS_BALL_CLS], verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None

    # 4) Pick highest-confidence box (already filtered to class 32)
    best = max(
        r.boxes,
        key=lambda b: float(b.conf[0])
    )
    x1, y1, x2, y2 = map(int, map(float, best.xyxy[0]))
    score = float(best.conf[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if show:
        draw = img.copy()
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(draw, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imshow("Ball", draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (cx, cy), (x1, y1, x2, y2, score)

def benchmark(image_path, runs=100):
    # Warmup once (build CUDA graph / cudnn autotune etc.)
    find_ball_yolo(image_path, show=True)
    t0 = time.perf_counter()
    ok = 0
    for _ in range(runs):
        res = find_ball_yolo(image_path, show=False)
        ok += int(res is not None)
    dt = time.perf_counter() - t0
    fps = runs / dt
    print(f"Runs: {runs}, Hits: {ok}, Total: {dt:.3f}s, FPS: {fps:.1f}")
    return fps

if __name__ == "__main__":
    start = time.time()
    out = find_ball_yolo("1.png", show=False)
    print("Time taken:", time.time() - start)
    if out:
        (x, y), (x1, y1, x2, y2, s) = out
        print(f"Ball center: ({x}, {y})  bbox=({x1},{y1},{x2},{y2})  score={s:.3f}")
    else:
        print("No ball detected.")
    benchmark("1.png", runs=120)

