"""
Standalone mAP measurement script.
Loads yolo26n.pt, runs full inference on video frames,
then computes mAP@50 using conf=0.25, iou=0.5 (current thresholds).

Usage:
    cd split_inference
    python measure_map.py
    python measure_map.py --frames 200 --model yolo26n
"""
import argparse
import os
import sys
import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.Model import inference, postprocess_yolo

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="yolo26n",          help="model name (without .pt)")
parser.add_argument("--video",  default="video.mp4",        help="input video")
parser.add_argument("--gt",     default="datasets/groundtruth", help="ground truth dir")
parser.add_argument("--frames", type=int, default=0,        help="max frames (0=all)")
parser.add_argument("--batch",  type=int, default=32,       help="batch size")
parser.add_argument("--conf",   type=float, default=0.25,   help="confidence threshold")
parser.add_argument("--iou",    type=float, default=0.5,    help="NMS IoU threshold")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── load model ────────────────────────────────────────────────────────────────
pt_path = f"{args.model}.pt"
if not os.path.exists(pt_path):
    print(f"[ERROR] {pt_path} not found. Run server first to download the model.")
    sys.exit(1)

print(f"[INFO] Loading {pt_path} on {DEVICE} ...")
ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)
yolo = ckpt["model"].to(DEVICE).float()
yolo.eval()
layers = list(yolo.model)
print(f"[INFO] Model loaded — {len(layers)} layers")

# ── load ground truth ─────────────────────────────────────────────────────────
gt_dict = {}
if os.path.isdir(args.gt):
    for fname in sorted(os.listdir(args.gt)):
        if not fname.endswith(".txt"):
            continue
        try:
            num = int(os.path.splitext(fname)[0].split("_")[-1])
        except ValueError:
            continue
        boxes, labels = [], []
        with open(os.path.join(args.gt, fname), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                boxes.append([(cx - bw/2)*640, (cy - bh/2)*640,
                               (cx + bw/2)*640, (cy + bh/2)*640])
                labels.append(int(cls))
        gt_dict[num] = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64)   if labels else torch.zeros(0, dtype=torch.int64),
        }
    print(f"[INFO] Loaded GT for {len(gt_dict)} frames from '{args.gt}'")
else:
    print(f"[WARN] GT dir '{args.gt}' not found — mAP will be skipped")

# ── torchmetrics ──────────────────────────────────────────────────────────────
try:
    from torchmetrics.detection import MeanAveragePrecision
    map_metric = MeanAveragePrecision(iou_type="bbox")
    map_metric.warn_on_many_detections = False
except ImportError:
    print("[ERROR] torchmetrics not installed: pip install torchmetrics")
    sys.exit(1)

# ── video read ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {args.video}")
    sys.exit(1)

total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_frames  = args.frames if args.frames > 0 else total_video
print(f"[INFO] Processing up to {max_frames} frames (video has {total_video})")
print(f"[INFO] conf={args.conf}  iou={args.iou}  batch={args.batch}")

frame_buf  = []
frame_nums = []
frame_count = 0

def process_batch(buf, nums):
    imgs = np.stack(buf, axis=0).astype(np.float32) / 255.0          # [B,H,W,3]
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(DEVICE)     # [B,3,H,W]
    with torch.no_grad():
        x, y = imgs, []
        x, _ = inference(layers, x, y, cut=0)
    # conf=0.001 → torchmetrics receives full PR curve (standard YOLO mAP eval)
    map_results = postprocess_yolo(x, conf_thres=0.001, iou_thres=args.iou)
    for r, fn in zip(map_results, nums):
        if fn not in gt_dict:
            continue
        map_metric.update(
            [{"boxes":  r["boxes"].cpu().float(),
              "scores": r["scores"].cpu().float(),
              "labels": r["classes"].cpu().long()}],
            [gt_dict[fn]],
        )

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_buf.append(frame)
    frame_nums.append(frame_count)

    if len(frame_buf) == args.batch:
        process_batch(frame_buf, frame_nums)
        frame_buf, frame_nums = [], []
        print(f"  processed {frame_count}/{max_frames} frames", end="\r")

if frame_buf:
    process_batch(frame_buf, frame_nums)

cap.release()
print(f"\n[INFO] Done. Processed {frame_count} frames.")

# ── compute mAP ───────────────────────────────────────────────────────────────
result = map_metric.compute()
map50    = float(result["map_50"])
map5095  = float(result["map"])

print(f"\n{'='*45}")
print(f"  mAP@50      = {map50:.4f}")
print(f"  mAP@50:95   = {map5095:.4f}")
print(f"{'='*45}")
print(f"\n  Current a0  = 0.28  (in CpcdrlController.py)")
print(f"  New mAP     = {map50:.4f}")

margin = 0.03
suggested_a0 = round(map50 - margin, 3)
if suggested_a0 < map50:
    print(f"  Suggested a0 = {suggested_a0:.3f}  (= mAP - {margin})")
    if suggested_a0 < 0.28:
        print(f"  [!] a0 cần giảm từ 0.28 → {suggested_a0:.3f} để reward không bị âm")
    else:
        print(f"  [OK] a0 hiện tại (0.28) vẫn OK (mAP còn cao hơn đủ)")
