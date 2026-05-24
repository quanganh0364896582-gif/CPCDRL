"""
Reset DQN weights + replay buffer while keeping DDPG intact.
Run once, then restart run_loop.ps1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from src.CpcdrlController import CpcdrlController
from src.CpcdrlAdapter import controller_save_path

EDGE_DEVICE = "DEVICE_7"
path = controller_save_path(EDGE_DEVICE)

if not Path(path).exists():
    print(f"No checkpoint found at {path}, nothing to reset.")
    sys.exit(0)

print(f"Loading checkpoint: {path}")
data = torch.load(path, map_location="cpu", weights_only=False)

# Build fresh controller (fresh DQN weights + empty replay)
ctrl = CpcdrlController(
    config         = data["config"],
    edge_device    = data.get("edge_device",    "DEVICE_7"),
    cloud_device   = data.get("cloud_device",   "DAI"),
    model_name     = data.get("model_name",     "yolo26n"),
    batch_size     = data.get("batch_size",     32),
    bandwidth_mbps = data.get("bandwidth_mbps", None),
)

# Restore DDPG weights (keep learned compression policy)
ctrl.agent_c.actor.load_state_dict(data["ddpg_actor"])
ctrl.agent_c.actor_target.load_state_dict(data["ddpg_actor_tgt"])
ctrl.agent_c.critic.load_state_dict(data["ddpg_critic"])
ctrl.agent_c.critic_target.load_state_dict(data["ddpg_critic_tgt"])
print("DDPG weights restored.")

# Fresh DQN: random init, epsilon=0.30, state_p=ones
ctrl.agent_p.epsilon = 0.30
ctrl.state_p = np.ones(24, dtype=np.float32)
# replay buffer stays empty (fresh start)
print("DQN reset: fresh weights, epsilon=0.30, state_p=ones.")

# Keep session context if mid-session
ctrl._session_cut_point   = data.get("session_cut_point",   None)
ctrl._session_ratios      = data.get("session_ratios",      None)
ctrl._session_ddpg_state0 = data.get("session_ddpg_state0", None)

ctrl.save(path)
print(f"Saved to {path}. Ready to restart training.")
