"""
profile_device.py
Đo per-layer timing + băng thông cho thiết bị thực.

Cách dùng:
  1. Trong config.yaml đặt profiling.enable: True và điền device_name
  2. Chạy: python profile_device.py
  3. Script in code sẵn để paste vào CpcdrlAdapter.py
  4. Đặt profiling.enable: False, train bình thường
"""
import sys
import time
import json
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import pika

# ── đọc config ─────────────────────────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

prof_cfg = config.get("profiling", {})

if not prof_cfg.get("enable", False):
    print("profiling.enable = False trong config.yaml. Đặt True rồi chạy lại.")
    sys.exit(0)

DEVICE_NAME = prof_cfg.get("device_name", "NEW_DEVICE").upper().replace(" ", "_")
NUM_RUNS    = int(prof_cfg.get("num_runs",  10))
WARMUP      = int(prof_cfg.get("warmup",    3))
BATCH_SIZE  = int(config["server"]["batch-size"])
MODEL_NAME  = config["server"]["model"]
N_LAYERS    = 24

print(f"\n=== Profiling {DEVICE_NAME} ===")
print(f"Model: {MODEL_NAME}, batch_size={BATCH_SIZE}, runs={NUM_RUNS}, warmup={WARMUP}\n")

# ── load model ─────────────────────────────────────────────────────────────────
print("[1/3] Loading model...")
from ultralytics import YOLO as _YOLO
_yolo        = _YOLO(f"{MODEL_NAME}.pt")
model_layers = _yolo.model.model

from src.Model import inference as _inference

dummy_input = torch.zeros(BATCH_SIZE, 3, 640, 640)

# ── đo per-layer timing ────────────────────────────────────────────────────────
print("[2/3] Measuring per-layer inference timing...")

cumulative_times = []
for cut in range(1, N_LAYERS + 1):
    run_times = []
    for run in range(WARMUP + NUM_RUNS):
        y = []
        x = dummy_input.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _inference(model_layers[:cut], x, y, 0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if run >= WARMUP:
            run_times.append(t1 - t0)
    cumulative_times.append(float(np.mean(run_times)))
    print(f"  cut={cut:2d}: {cumulative_times[-1]*1000:.2f}ms", end="\r")

print(f"  Done. Tổng edge (all layers): {cumulative_times[-1]*1000:.1f}ms{' '*20}")

per_layer_timing = [cumulative_times[0]]
for i in range(1, N_LAYERS):
    per_layer_timing.append(max(0.0, cumulative_times[i] - cumulative_times[i-1]))

# ── đo băng thông ──────────────────────────────────────────────────────────────
print("[3/3] Measuring network bandwidth via RabbitMQ...")
rabbit  = config["rabbit"]
bw_mbps = None

try:
    credentials = pika.PlainCredentials(rabbit["username"], rabbit["password"])
    conn = pika.BlockingConnection(pika.ConnectionParameters(
        rabbit["address"], 5672, rabbit["virtual-host"], credentials,
        socket_timeout=10,
    ))
    ch     = conn.channel()
    test_q = f"profiling_bw_{DEVICE_NAME}"
    ch.queue_declare(queue=test_q, auto_delete=True)

    msg       = pickle.dumps({"data": bytes(int(10 * 1024 * 1024))})  # 10 MB
    actual_mb = len(msg) / (1024 * 1024)
    samples   = []
    for _ in range(5):
        t0 = time.perf_counter()
        ch.basic_publish(exchange="", routing_key=test_q, body=msg)
        for method, _, _ in ch.consume(test_q, inactivity_timeout=5):
            ch.basic_ack(method.delivery_tag)
            break
        samples.append(actual_mb / (time.perf_counter() - t0))

    bw_mbps = float(np.median(samples))
    ch.queue_delete(queue=test_q)
    conn.close()
    print(f"  Bandwidth: {bw_mbps:.1f} MB/s  ({bw_mbps*8/1000:.3f} Gbps)")

except Exception as e:
    bw_mbps = 115.0
    print(f"  RabbitMQ không khả dụng ({e}). Dùng mặc định 115 MB/s")

link_gbps = round(bw_mbps * 8 / 1000, 3)

# ── lưu JSON backup ────────────────────────────────────────────────────────────
out_path = f"profile_{DEVICE_NAME}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "device_name":      DEVICE_NAME,
        "per_layer_timing": per_layer_timing,
        "bandwidth_mbps":   bw_mbps,
        "link_gbps":        link_gbps,
    }, f, indent=2)
print(f"\n  Saved backup: {out_path}")

# ── tự động ghi vào devices.yaml ──────────────────────────────────────────────
devices_path = Path("devices.yaml")
devices_data = yaml.safe_load(devices_path.read_text(encoding="utf-8")) or {}

if DEVICE_NAME in devices_data:
    print(f"\n  {DEVICE_NAME} đã có trong devices.yaml — ghi đè.")

devices_data[DEVICE_NAME] = {
    "cores":     prof_cfg.get("cores"),
    "gflops":    prof_cfg.get("gflops"),
    "link_gbps": link_gbps,
    "timing":    [round(v, 6) for v in per_layer_timing],
}

# Ghi lại với comment header giữ nguyên
header = ""
raw_text = devices_path.read_text(encoding="utf-8")
for line in raw_text.splitlines():
    if line.startswith("#"):
        header += line + "\n"
    else:
        break

with open(devices_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for dev, vals in devices_data.items():
        timing_inline = "[" + ", ".join(str(v) for v in vals["timing"]) + "]"
        f.write(f"{dev}:\n")
        f.write(f"  cores: {vals['cores']}\n")
        f.write(f"  gflops: {vals['gflops']}\n")
        f.write(f"  link_gbps: {vals['link_gbps']}\n")
        f.write(f"  timing: {timing_inline}\n\n")

print(f"  devices.yaml updated — {DEVICE_NAME} added.")
if devices_data[DEVICE_NAME]["cores"] is None:
    print("  Lưu y: cores/gflops chưa điền. Sửa trực tiếp trong devices.yaml nếu cần.")
print()
print("=== Done. Đặt profiling.enable: False rồi train bình thường. ===")
