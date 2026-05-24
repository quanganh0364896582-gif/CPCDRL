"""
CpcdrlController.py
Session-level RL controller for split_inference, faithful to the CPCDRL paper.

Algorithm (Section III of the paper, Figure 2 "ping-pong"):
  - AgentAp (DQN):  state = compression_ratios (24-dim), action = cut_point (0-24)
  - AgentAc (DDPG): given cut_point, generates per-layer ratio for each EDGE layer

One inference run (full video) = one episode.

Session workflow
----------------
  controller = CpcdrlController.load()   # or fresh init
  cut_int, num_bits = controller.decide()
  ... run inference ...
  controller.observe(e2e_latency_ms, map50)
  controller.save()
"""
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# ── locate cpcdrl_baseline ────────────────────────────────────────────────────
_THIS      = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]   # d:\hungarian
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from cpcdrl_baseline import (
    CPCDRLConfig,
    AgentAp_DQN,
    AgentAc_DDPG,
)
from src.CpcdrlAdapter import (
    N_YOLO_LAYERS,
    DEVICE_CATALOGUE,
    get_device_timing,
    get_device_bandwidth_mbps,
    make_real_profile,
    compute_latency_real,
    ratios_to_bits_per_layer,
    mean_ratio_to_num_bits,
    resolve_model_key,
    controller_save_path,
)

# Valid split range for split_inference (edge and cloud both non-empty)
_MIN_CUT = 1
_MAX_CUT = N_YOLO_LAYERS - 1   # 23

DEFAULT_CONFIG = CPCDRLConfig(
    a0=0.28,         # below real mAP (~0.325) so exp term > 1
    eta=5.0,         # accuracy sensitivity
    mu=1.0,          # latency penalty weight — paper value, paired with estimated latency (0.1-0.4s)
    episodes=500,    # not used directly here
    epsilon=0.30,    # DQN initial exploration
    gamma=0.99,
    replay_size=500,
    batch_size=16,
    sigma=0.10,      # DDPG exploration noise (Eq. 12)
    seed=42,
)


class CpcdrlController:
    """
    Session-level coordinator that wraps AgentAp_DQN and AgentAc_DDPG.

    State carried across sessions
    -----------------------------
    state_p: np.ndarray, shape (N_YOLO_LAYERS,)
        Compression ratios from the last DDPG call — this is what the paper
        uses as DQN's state (Section III-A-1, "next state s_p = a_c").
        Initialised to all-ones (no pruning) on the first session.
    """

    def __init__(
        self,
        config:        CPCDRLConfig = DEFAULT_CONFIG,
        edge_device:   str          = "DEVICE_1",
        cloud_device:  str          = "DAI",
        model_name:    str          = "yolo26n",
        batch_size:    int          = 32,
        bandwidth_mbps: float       = None,
    ):
        self.config        = config
        self.edge_device   = edge_device
        self.cloud_device  = cloud_device
        self.model_name    = model_name
        self.batch_size    = batch_size
        self.model_key     = resolve_model_key(model_name, batch_size)
        self.edge_timing   = get_device_timing(edge_device)
        self.cloud_timing  = get_device_timing(cloud_device)
        if bandwidth_mbps is None:
            bandwidth_mbps = get_device_bandwidth_mbps(edge_device)
        self.bandwidth_mbps = bandwidth_mbps

        # Real hardware profile used by DDPG._build_layer_state()
        self.profile = make_real_profile(
            edge_device    = edge_device,
            model_key      = self.model_key,
            bandwidth_mbps = bandwidth_mbps,
        )

        # RL agents — identical to run_cpcdrl_training() in cpcdrl_baseline.py
        self.agent_p = AgentAp_DQN(N_YOLO_LAYERS, config)   # DQN: cut-point selector
        self.agent_c = AgentAc_DDPG(N_YOLO_LAYERS, config)  # DDPG: per-layer compression

        # DQN inter-session state (paper: "s_p = last compression ratios")
        self.state_p: np.ndarray = np.ones(N_YOLO_LAYERS, dtype=np.float32)

        # Saved at decide() for use in observe()
        self._session_cut_point:       Optional[int]          = None
        self._session_ratios:          Optional[np.ndarray]   = None  # (N,) DDPG output
        self._session_ddpg_state0:     Optional[np.ndarray]   = None  # 5-dim state at layer 0

    # ── public API ────────────────────────────────────────────────────────────

    def decide(self) -> Tuple[int, list]:
        """
        Run one step of the CPCDRL ping-pong for the upcoming session.

        Order (Figure 2):
          1. DQN selects cut_point from state_p (= last DDPG ratios)
          2. DDPG generates per-layer compression ratios for edge layers
          3. Per-layer ratio -> per-layer num_bits (paper-faithful, not averaged)

        Returns
        -------
        cut_int        : layer index to split at (1-23)
        bits_per_layer : list of int, length=cut_int; quantisation bits for each
                         edge layer i, derived from DDPG ratio_i continuously
        """
        # Step 1 — DQN picks cut point (action space: 0 .. N_YOLO_LAYERS)
        raw_cut = self.agent_p.select_action(self.state_p)
        cut_int = int(np.clip(raw_cut, _MIN_CUT, _MAX_CUT))

        # Step 2 — DDPG generates per-layer ratios for edge layers 0..cut_int-1
        compression_ratios = self.agent_c.generate_compression_ratios(
            cut_int, self.profile
        )
        # compression_ratios[i] in (0.05, 1.0) for i < cut_int, 1.0 for i >= cut_int

        # Step 3 — per-layer bits: each layer i gets its own bit-depth from ratio_i
        bits_per_layer = ratios_to_bits_per_layer(compression_ratios, cut_int)

        # Save context for observe()
        self._session_cut_point   = cut_int
        self._session_ratios      = compression_ratios
        self._session_ddpg_state0 = self.agent_c._build_layer_state(
            layer_idx  = 0,
            profile    = self.profile,
            cut_point  = cut_int,
            prev_ratio = 1.0,
        )

        return cut_int, bits_per_layer

    def observe(self, e2e_latency_ms: float, map50: float) -> None:
        """
        Feed back real measurements from the completed session.

        Computes the shared reward, stores one transition per agent (episodic),
        and triggers one gradient update for each agent.

        Also updates state_p <- compression_ratios (paper: next state for DQN).
        """
        if self._session_cut_point is None:
            # No prior decide() — skip training this round
            return

        cut_point        = self._session_cut_point
        ratios           = self._session_ratios          # shape (N,)
        ddpg_state0      = self._session_ddpg_state0     # shape (5,)

        # ── Reward (Eq. 10) using ESTIMATED latency (faithful to paper) ─────────
        # Paper uses model-estimated latency (no system overhead). Real e2e
        # latency includes ~4-5s constant overhead (RabbitMQ, startup) that
        # drowns out the cut-point signal, so we use the profile estimate here.
        e2e_latency_ms = min(e2e_latency_ms, 12000.0)   # clamp spikes (for logging)
        est_latency_s = compute_latency_real(
            cut_point          = cut_point,
            compression_ratios = ratios,
            edge_timing        = self.edge_timing,
            model_key          = self.model_key,
            bandwidth_mbps     = self.bandwidth_mbps,
            cloud_timing       = self.cloud_timing,
        )
        mean_ratio  = float(np.mean(ratios[:cut_point])) if cut_point > 0 else 1.0
        alpha       = 1.0 - mean_ratio   # fraction pruned
        reward      = (
            float(np.exp(self.config.eta * (map50 - self.config.a0)))
            * (1.0 - alpha)
            - self.config.mu * est_latency_s
        )

        # ── Next state for DQN = ratios from this session (paper Sec. III-A-1) ──
        next_state_p = ratios.astype(np.float32)

        # ── Next state for DDPG critic (paper uses layer cut_point state) ───────
        next_ddpg_state = self.agent_c._build_layer_state(
            layer_idx  = min(cut_point, N_YOLO_LAYERS - 1),
            profile    = self.profile,
            cut_point  = cut_point,
            prev_ratio = float(ratios[0]),
        )

        # ── Store transitions (one per episode, matching cpcdrl_baseline.py) ────
        self.agent_p.store(self.state_p, cut_point, reward, next_state_p)
        self.agent_c.store(ddpg_state0, ratios, reward, next_ddpg_state)

        # ── Gradient updates ──────────────────────────────────────────────────
        self.agent_p.update()
        self.agent_c.update()

        # ── Decay DQN epsilon ────────────────────────────────────────────────
        self.agent_p.epsilon = max(0.05, self.agent_p.epsilon * 0.99)

        # ── Advance state_p for next session ─────────────────────────────────
        self.state_p = next_state_p

        # ── Append to training log ────────────────────────────────────────────
        try:
            import csv, os
            log_path = f"training_log_{self.edge_device}.csv"
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["episode", "cut_point", "e2e_ms", "map50", "reward", "epsilon"])
                episode = sum(1 for _ in open(log_path)) - 1  # count rows - header
                w.writerow([episode, cut_point, round(e2e_latency_ms, 1),
                            round(map50, 4), round(reward, 4),
                            round(self.agent_p.epsilon, 4)])
        except Exception:
            pass

        # Reset session context
        self._session_cut_point   = None
        self._session_ratios      = None
        self._session_ddpg_state0 = None

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None) -> None:
        if path is None:
            path = controller_save_path(self.edge_device)
        torch.save({
            "config":              self.config,
            "edge_device":         self.edge_device,
            "cloud_device":        self.cloud_device,
            "model_name":          self.model_name,
            "batch_size":          self.batch_size,
            "bandwidth_mbps":      self.bandwidth_mbps,
            "state_p":             self.state_p,
            "dqn_epsilon":         self.agent_p.epsilon,
            "dqn_policy":          self.agent_p.policy_net.state_dict(),
            "dqn_target":          self.agent_p.target_net.state_dict(),
            "ddpg_actor":          self.agent_c.actor.state_dict(),
            "ddpg_actor_tgt":      self.agent_c.actor_target.state_dict(),
            "ddpg_critic":         self.agent_c.critic.state_dict(),
            "ddpg_critic_tgt":     self.agent_c.critic_target.state_dict(),
            # session context — needed so observe() works after server restart
            "session_cut_point":   self._session_cut_point,
            "session_ratios":      self._session_ratios,
            "session_ddpg_state0": self._session_ddpg_state0,
            # replay buffers — preserved across restarts so cut=8/9/11 experiences survive
            "dqn_replay": [
                (t.state.tolist(), int(t.action), float(t.reward), t.next_state.tolist())
                for t in self.agent_p.memory._buf
            ],
        }, path)

    @classmethod
    def load(cls, path: str) -> "CpcdrlController":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj  = cls(
            config         = data["config"],
            edge_device    = data.get("edge_device",  "DEVICE_1"),
            cloud_device   = data.get("cloud_device", "DAI"),
            model_name     = data.get("model_name",   "yolo26n"),
            batch_size     = data.get("batch_size",   32),
            bandwidth_mbps = data.get("bandwidth_mbps", None),
        )
        obj.state_p               = data.get("state_p", obj.state_p)
        obj.agent_p.epsilon       = data.get("dqn_epsilon", obj.agent_p.epsilon)
        obj.agent_p.policy_net.load_state_dict(data["dqn_policy"])
        obj.agent_p.target_net.load_state_dict(data["dqn_target"])
        obj.agent_c.actor.load_state_dict(data["ddpg_actor"])
        obj.agent_c.actor_target.load_state_dict(data["ddpg_actor_tgt"])
        obj.agent_c.critic.load_state_dict(data["ddpg_critic"])
        obj.agent_c.critic_target.load_state_dict(data["ddpg_critic_tgt"])
        obj._session_cut_point    = data.get("session_cut_point",   None)
        obj._session_ratios       = data.get("session_ratios",      None)
        obj._session_ddpg_state0  = data.get("session_ddpg_state0", None)
        for s, a, r, ns in data.get("dqn_replay", []):
            obj.agent_p.memory.push(
                np.array(s, dtype=np.float32), a, r, np.array(ns, dtype=np.float32)
            )
        return obj
