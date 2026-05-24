# Split Inference with CPCDRL

Distributed YOLO inference across edge and cloud devices, with an RL controller that adaptively selects the **cut point** (where to split the model) and **per-layer quantisation bits** each session.

<p align="center">
  <img src="imgs/overview.png" width="850">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [CPCDRL Adaptive Controller](#cpcdrl-adaptive-controller)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [CPCDRL Training Loop](#cpcdrl-training-loop)
- [Profiling a New Device](#profiling-a-new-device)
- [Device Profiles](#device-profiles)
- [Supported Models](#supported-models)

---

## Overview

Instead of transmitting raw video frames to a server, the edge device runs the **first N layers** of the YOLO model and sends only the **intermediate feature maps** (compressed) to the cloud, which runs the remaining layers and returns detections.

This reduces bandwidth and improves scalability, at the cost of a tunable trade-off between edge compute, transmission size, and cloud compute — controlled by the **cut point** and **quantisation bits**.

---

## Architecture

<p align="center">
  <img src="imgs/SI-Inference.jpg" width="900">
</p>

### Components

| Component | Role |
|-----------|------|
| **Edge client** (`client.py --layer_id 1`) | Runs YOLO layers `0..cut-1`, compresses output, sends to cloud |
| **Cloud client** (`client.py --layer_id 2`) | Receives feature maps, runs layers `cut..23`, returns detections |
| **Server** (`server.py`) | Registers clients, issues cut point / compression decisions, collects feedback |
| **RabbitMQ** | Message broker between all components |
| **CPCDRL controller** (`src/CpcdrlController.py`) | RL agent that adaptively chooses cut point and bits each session |

### Message flow

```
Edge  ──[feature maps]──►  RabbitMQ  ──►  Cloud
Edge  ──[NOTIFY]────────►  Server
Server ──[STOP]──────────►  Edge, Cloud
Server ──[feedback read]─►  Controller ──[decide()]──► next session
```

---

## CPCDRL Adaptive Controller

Based on the paper *"Collaborative DNNs Inference with Joint Model Partition and Compression in Mobile Edge-Cloud Computing Networks"* (IEEE WCNC 2024).

### Algorithm

Two RL agents operate in a **ping-pong** loop each session:

1. **AgentAp (DQN)** — selects the cut point (layer index to split at, 1–23)
2. **AgentAc (DDPG)** — generates a per-layer compression ratio for each edge layer

The per-layer ratio maps to a **quantisation bit-depth** (4–16 bits) applied during inference.

### Reward (Eq. 10)

```
R = exp(η × (mAP50 - A0)) × (1 - α) - μ × T_est
```

| Symbol | Value | Meaning |
|--------|-------|---------|
| η | 5.0 | Accuracy sensitivity |
| A0 | 0.28 | Baseline mAP threshold |
| α | 1 − mean_ratio | Fraction of data pruned |
| μ | 1.0 | Latency penalty weight |
| T_est | seconds | Profile-estimated E2E latency (not real, avoids startup overhead) |

### Session workflow

```
Server starts
  └─ load controller checkpoint  (cpcdrl_controller_DEVICE_X.pt)
  └─ read last-session feedback  (cpcdrl_feedback_DEVICE_X.json)
  └─ observe(e2e_ms, mAP)        → DQN + DDPG gradient update
  └─ decide()                    → cut_int, bits_per_layer
  └─ send START to edge client

Edge runs inference with chosen cut + bits
  └─ writes cpcdrl_feedback_DEVICE_X.json

Next server start repeats the loop
```

### Training logs

Each device writes `training_log_DEVICE_X.csv`:

```
episode, cut_point, e2e_ms, map50, reward, epsilon
```

---

## Project Structure

```
split_inference/
├── server.py                        # Central controller + CPCDRL init
├── client.py                        # Edge or cloud inference node
├── config.yaml                      # All runtime configuration
├── devices.yaml                     # Hardware profiles (timing, bandwidth)
├── run_loop.ps1                     # PowerShell script to run N episodes
├── profile_device.py                # Measure per-layer timing + bandwidth for a new device
├── reset_dqn.py                     # Reset DQN weights while keeping DDPG
├── requirements.txt
│
├── src/
│   ├── CpcdrlController.py          # Session-level RL coordinator (DQN + DDPG)
│   ├── CpcdrlAdapter.py             # Bridge: real measurements ↔ CPCDRL formulation
│   ├── Scheduler.py                 # Inference pipeline (first_layer / last_layer)
│   ├── Compress.py                  # Quantisation delta codec (Encoder / Decoder)
│   ├── Model.py                     # YOLO layer runner + per-layer quantisation
│   ├── Server.py                    # RabbitMQ server class
│   ├── RpcClient.py                 # RabbitMQ client class
│   ├── Log.py                       # Coloured console logging
│   └── Utils.py                     # RabbitMQ queue cleanup, mAP helpers
│
└── imgs/
    ├── overview.png
    ├── SI-Inference.jpg
    └── START.png
```

**Runtime artefacts** (generated, not committed):

| File | Description |
|------|-------------|
| `cpcdrl_controller_DEVICE_X.pt` | Controller checkpoint (DQN + DDPG weights, replay buffer) |
| `cpcdrl_feedback_DEVICE_X.json` | Last-session feedback waiting to be observed |
| `training_log_DEVICE_X.csv` | Per-episode training history |
| `profile_DEVICE_X.json` | Raw profiling backup |
| `metrics_pivoted.csv` | Last-run per-batch metrics (edge + cloud joined) |
| `detections.json` | Per-frame detection results |

---

## Setup

### Requirements

- Python 3.8+
- RabbitMQ (with management plugin)

### Install

```bash
git clone https://github.com/filrg/split_inference
cd split_inference
pip install -r requirements.txt
```

### Start RabbitMQ

RabbitMQ must be running before starting any component.

Admin UI: `http://localhost:15672` (default credentials: `guest` / `guest`)

---

## Configuration

All settings are in `config.yaml`:

```yaml
name: YOLO
server:
  cut-layer: a          # static fallback cut: a=4, b=11, c=17, d=23
  clients:
    - 1                 # number of edge clients
    - 1                 # number of cloud clients
  model: yolo26n        # model name (must match a .pt file)
  batch-size: 32

rabbit:
  address: 127.0.0.1
  username: guest
  password: guest
  virtual-host: /

debug-mode: False
data: video.mp4
max_frames: 160         # frames per episode (0 = full video)
log-path: .

compress:
  enable: True
  num_bit: 8            # default quantisation bits (overridden by CPCDRL when enabled)

cpcdrl:
  enable: False         # set True to activate RL controller
  edge_device: ["DEVICE_4"]   # one or more edge hardware profiles from devices.yaml
  cloud_device: ["DAI"]       # cloud hardware profile

profiling:
  enable: False         # set True then run profile_device.py
  device_name: Personal
  cores: null           # fill manually
  gflops: null          # fill manually
  num_runs: 10
  warmup: 3
```

### clients array

When running multiple simultaneous edge devices, set `clients: [N, 1]` where N = number of edge devices listed in `edge_device`.

---

## Running the System

### Manual (3 terminals)

**Terminal 1 — Server:**
```bash
python server.py
```

**Terminal 2 — Edge client:**
```bash
python client.py --layer_id 1 --edge_device DEVICE_4
```

**Terminal 3 — Cloud client:**
```bash
python client.py --layer_id 2
```

### Automated loop (PowerShell)

```powershell
# Run 200 episodes (reads edge_device from config.yaml)
.\run_loop.ps1 -N 200
```

The script starts server + one edge process per device in `edge_device` + one cloud process, waits for completion, then repeats.

---

## CPCDRL Training Loop

### Start training

1. Set `cpcdrl.enable: True` in `config.yaml`
2. Set `edge_device` to the target device(s)
3. Set `clients` count to match number of edge devices
4. Run:

```powershell
.\run_loop.ps1 -N 200
```

### Monitor progress

```powershell
Get-Content training_log_DEVICE_4.csv | Select-Object -Last 10
```

Or watch live:
```powershell
Get-Content training_log_DEVICE_4.csv -Wait -Tail 5
```

### Reset DQN only (keep DDPG)

If the DQN gets stuck, reset its weights while preserving the learned compression policy:

```bash
# Edit EDGE_DEVICE in reset_dqn.py first
python reset_dqn.py
```

### Training multiple devices simultaneously

```yaml
cpcdrl:
  enable: True
  edge_device: ["DEVICE_2", "DEVICE_4", "DEVICE_7"]
server:
  clients:
    - 3
    - 1
```

```powershell
.\run_loop.ps1 -N 200
```

> **Note:** When running multiple edge devices on the same machine, e2e latency values will be inflated due to CPU contention. Training is unaffected because the reward uses profile-estimated latency, not real e2e.

---

## Profiling a New Device

To add a real hardware device, measure its per-layer timing and network bandwidth:

1. In `config.yaml`, set:
   ```yaml
   profiling:
     enable: True
     device_name: MY_DEVICE   # will be normalised to upper-case
     cores: 4                 # fill in manually
     gflops: 70.0             # fill in manually
   ```

2. Run on the target device:
   ```bash
   python profile_device.py
   ```

3. The script:
   - Measures per-layer inference timing (24 values)
   - Measures RabbitMQ round-trip bandwidth
   - Saves backup `profile_MY_DEVICE.json`
   - Automatically appends the entry to `devices.yaml`

4. Set `profiling.enable: False`, then use `edge_device: ["MY_DEVICE"]` in the `cpcdrl` section.

---

## Device Profiles

`devices.yaml` stores hardware profiles for all known devices. It is read at startup by `CpcdrlAdapter.py`.

```yaml
DEVICE_4:
  cores: 2
  gflops: 47.45
  link_gbps: 0.922          # upload bandwidth to cloud
  timing: [0.002141, ...]   # per-layer inference time (seconds), 24 values
```

Pre-profiled devices: `DAI` (cloud server), `DEVICE_1` through `DEVICE_9`.

`link_gbps: null` means the device is the cloud (no upload).

---

## Supported Models

Cut-point tensor sizes are embedded for the following model+batch combinations:

| Model | Batch sizes |
|-------|-------------|
| yolo26n | 32, 48, 64 |
| yolo26m | 32 |
| yolo26l | 32 |
| yolo26x | 32, 48, 64 |
| yolo11n | 32, 48, 64 |
| yolo11m | 32 |
| yolo11l | 32 |
| yolo11x | 32, 48, 64 |

The model file must exist as `<model_name>.pt` in the working directory (downloaded automatically on first run via Ultralytics).

---

## License

See [LICENSE](./LICENSE)
