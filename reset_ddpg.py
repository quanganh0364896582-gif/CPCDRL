"""
Reset DDPG weights (actor + critic + replay) while keeping DQN intact.

The old DDPG was trained incorrectly (only layer-0 state, mean action).
After fixing cpcdrl_baseline.py to store per-layer transitions, run this
script once to clear the biased DDPG weights before restarting training.

DQN is preserved: cut-point policy, epsilon, state_p, replay buffer.

Usage (run from split_inference/):
    python reset_ddpg.py                         # resets all found checkpoints
    python reset_ddpg.py DEVICE_4 DEVICE_7       # resets specific devices
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from src.CpcdrlController import CpcdrlController
from src.CpcdrlAdapter import controller_save_path

devices = sys.argv[1:] if len(sys.argv) > 1 else None

if devices is None:
    # Auto-discover all checkpoints in current directory
    devices = [
        p.stem.replace("cpcdrl_controller_", "")
        for p in Path(".").glob("cpcdrl_controller_*.pt")
    ]
    if not devices:
        print("No cpcdrl_controller_*.pt files found.")
        sys.exit(0)

for dev in devices:
    path = controller_save_path(dev)
    if not Path(path).exists():
        print(f"[{dev}] No checkpoint at {path} — skipped.")
        continue

    print(f"[{dev}] Loading {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    ctrl = CpcdrlController(
        config         = data["config"],
        edge_device    = data.get("edge_device",    dev),
        cloud_device   = data.get("cloud_device",   "DAI"),
        model_name     = data.get("model_name",     "yolo26n"),
        batch_size     = data.get("batch_size",     32),
        bandwidth_mbps = data.get("bandwidth_mbps", None),
    )

    # Restore DQN (cut-point policy stays intact)
    ctrl.agent_p.policy_net.load_state_dict(data["dqn_policy"])
    ctrl.agent_p.target_net.load_state_dict(data["dqn_target"])
    ctrl.agent_p.epsilon = data.get("dqn_epsilon", 0.05)
    ctrl.state_p = data.get("state_p", np.ones(24, dtype=np.float32))
    for s, a, r, ns in data.get("dqn_replay", []):
        ctrl.agent_p.memory.push(
            np.array(s, dtype=np.float32), a, r, np.array(ns, dtype=np.float32)
        )
    print(f"[{dev}] DQN restored (epsilon={ctrl.agent_p.epsilon:.3f}, "
          f"replay={len(ctrl.agent_p.memory)} transitions).")

    # DDPG: fresh random weights, empty replay (old transitions were wrong)
    print(f"[{dev}] DDPG reset to random weights.")

    # Session context
    ctrl._session_cut_point = data.get("session_cut_point", None)
    ctrl._session_ratios    = data.get("session_ratios",    None)

    ctrl.save(path)
    print(f"[{dev}] Saved to {path}.\n")

print("Done. Restart run_loop to begin retraining DDPG with per-layer transitions.")
