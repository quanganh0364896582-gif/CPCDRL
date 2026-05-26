"""
Reset cả DQN và DDPG về random weights, xóa training log cũ.
Dùng trước khi train lại từ đầu.

Usage (chạy từ split_inference/):
    python reset_all.py                          # reset tất cả devices tìm được
    python reset_all.py DEVICE_2 DEVICE_4 DEVICE_7
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.CpcdrlController import CpcdrlController
from src.CpcdrlAdapter import controller_save_path, normalize_device_name

devices = sys.argv[1:] if len(sys.argv) > 1 else None

if devices is None:
    devices = [
        p.stem.replace("cpcdrl_controller_", "")
        for p in Path(".").glob("cpcdrl_controller_*.pt")
    ]
    if not devices:
        print("No cpcdrl_controller_*.pt files found.")
        sys.exit(0)

for dev in devices:
    dev = normalize_device_name(dev)
    path = controller_save_path(dev)

    if not Path(path).exists():
        print(f"[{dev}] No checkpoint at {path} — skipped.")
        continue

    print(f"[{dev}] Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Fresh controller — both DQN and DDPG get random weights
    ctrl = CpcdrlController(
        config         = data["config"],
        edge_device    = data.get("edge_device",    dev),
        cloud_device   = data.get("cloud_device",   "DAI"),
        model_name     = data.get("model_name",     "yolo26n"),
        batch_size     = data.get("batch_size",     32),
        bandwidth_mbps = data.get("bandwidth_mbps", None),
    )
    # Reset epsilon and state_p to initial values
    ctrl.agent_p.epsilon = ctrl.config.epsilon  # e.g. 0.30
    print(f"[{dev}] DQN + DDPG reset to random weights. epsilon={ctrl.agent_p.epsilon:.2f}")

    # Delete old training log so CSV starts fresh with new header
    log_path = Path(f"training_log_{dev}.csv")
    if log_path.exists():
        log_path.unlink()
        print(f"[{dev}] Deleted {log_path}")

    ctrl.save(path)
    print(f"[{dev}] Saved fresh checkpoint to {path}.\n")

print("Done. Run train_devices.ps1 / train_devices.sh to start training.")
