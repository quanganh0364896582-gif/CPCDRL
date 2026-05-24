"""
CpcdrlAdapter.py
Bridge between split_inference's real measurements and CPCDRL's formulation.

Device profiles (timing + catalogue) are loaded from devices.yaml (same folder
as config.yaml). Add new devices by running profile_device.py or editing
devices.yaml directly.
"""
import os
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Optional

# ── load device profiles from devices.yaml ───────────────────────────────────
_DEVICES_YAML = Path(__file__).resolve().parents[1] / "devices.yaml"

def _load_devices():
    with open(_DEVICES_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    timing   = {}
    catalogue = {}
    for name, vals in raw.items():
        timing[name] = np.array(vals["timing"], dtype=float)
        catalogue[name] = {
            "cores":     vals.get("cores"),
            "gflops":    vals.get("gflops"),
            "link_gbps": vals.get("link_gbps"),
        }
    return timing, catalogue

_TIMING_BY_DEVICE, DEVICE_CATALOGUE = _load_devices()

# ── cut-point tensor sizes (MB, 8-bit baseline, per model+batch_size) ─────────
CUT_DATA_SIZES_MB_BY_MODEL = {
    "yolo26n_bs32": np.array([
        43.97, 22.76, 30.89, 12.04, 25.18, 31.6,  31.33, 34.18, 34.27,
        34.34, 33.93, 44.32, 49.65, 40.29, 65.74, 88.87, 52.59, 55.39,
        61.56, 58.23, 59.52, 62.02, 59.83,
    ], dtype=float),
    "yolo26n_bs48": np.array([
        65.83, 34.14, 45.88, 17.81, 37.87, 47.5,  47.12, 51.37, 51.51,
        51.62, 51.02, 66.6,  74.85, 60.6,  98.91, 133.68, 79.04, 83.24,
        92.52, 87.48, 89.43, 93.2,  89.92,
    ], dtype=float),
    "yolo26n_bs64": np.array([
        87.3,  45.43, 61.03, 24.03, 50.56, 63.42, 62.89, 68.56, 68.73,
        68.9,  68.06, 88.74, 99.67, 80.78, 131.67, 178.36, 105.38, 110.99,
        123.36, 116.62, 119.2, 124.22, 119.8,
    ], dtype=float),
    "yolo26m_bs32": np.array([
        163.49, 68.03, 111.12, 42.36, 87.97, 110.87, 111.52, 116.96, 117.3,
        117.59, 116.03, 134.07, 157.56, 138.61, 228.94, 295.25, 188.05, 200.14,
        222.55, 208.7, 214.12, 218.42, 212.56,
    ], dtype=float),
    "yolo26l_bs32": np.array([
        163.34, 69.74, 139.46, 46.41, 96.33, 118.95, 120.97, 127.01, 126.86,
        127.28, 126.23, 147.28, 170.89, 150.25, 246.35, 319.53, 201.02, 214.07,
        237.26, 224.38, 230.2, 234.9, 228.58,
    ], dtype=float),
    "yolo26x_bs32": np.array([
        245.24, 92.61, 197.92, 65.87, 119.46, 148.81, 151.63, 159.68, 159.07,
        159.46, 158.35, 185.21, 211.14, 192.3, 328.09, 402.93, 261.99, 280.2,
        313.62, 294.44, 302.45, 308.58, 300.3,
    ], dtype=float),
    "yolo26x_bs48": np.array([
        367.06, 137.69, 297.61, 98.7, 178.67, 222.62, 226.94, 239.01, 238.08,
        238.67, 237.03, 277.37, 316.16, 288.51, 494.42, 600.83, 392.51, 419.77,
        469.75, 441.09, 453.1, 462.24, 449.86,
    ], dtype=float),
    "yolo26x_bs64": np.array([
        487.44, 183.03, 402.01, 129.05, 237.72, 296.36, 301.98, 318.03, 316.79,
        317.61, 315.36, 368.9, 420.39, 383.44, 655.76, 795.94, 522.23, 558.8,
        626.01, 587.01, 603.0, 615.11, 598.64,
    ], dtype=float),
    "yolo11n_bs32": np.array([
        45.41, 22.48, 31.13, 11.11, 24.92, 31.11, 31.01, 33.74, 34.01,
        33.91, 33.84, 45.18, 49.94, 40.21, 65.67, 87.91, 53.12, 56.15,
        62.46, 59.42, 60.89, 63.7,  61.97,
    ], dtype=float),
    "yolo11n_bs48": np.array([
        67.99, 33.72, 46.35, 16.7,  37.25, 46.56, 46.31, 50.4,  50.8,
        50.6,  50.57, 67.61, 74.25, 60.13, 98.37, 131.31, 79.31, 83.85,
        93.29, 88.76, 90.97, 95.2,  92.58,
    ], dtype=float),
    "yolo11n_bs64": np.array([
        90.55, 44.9,  61.69, 22.17, 49.52, 61.89, 61.59, 67.04, 67.59,
        67.42, 67.27, 89.99, 99.15, 80.01, 130.96, 174.56, 105.45, 111.48,
        124.05, 117.75, 120.69, 126.33, 122.83,
    ], dtype=float),
    "yolo11m_bs32": np.array([
        164.94, 61.35, 128.57, 48.34, 98.92, 123.57, 122.65, 128.7, 128.73,
        127.99, 127.94, 149.09, 171.76, 152.68, 251.62, 337.79, 203.13, 215.62,
        239.49, 226.61, 232.61, 237.68, 232.19,
    ], dtype=float),
    "yolo11l_bs32": np.array([
        165.3, 58.44, 159.0, 49.96, 93.33, 115.9, 115.59, 121.29, 121.31,
        119.81, 119.21, 133.72, 155.71, 141.74, 231.84, 306.93, 191.61, 203.84,
        226.15, 210.73, 216.42, 220.01, 215.48,
    ], dtype=float),
    "yolo11x_bs32": np.array([
        211.09, 94.66, 215.44, 69.23, 130.0, 161.9, 164.4, 172.97, 172.84,
        171.98, 170.89, 196.88, 228.61, 204.84, 340.64, 439.95, 279.74, 298.59,
        330.19, 313.12, 322.66, 329.11, 321.43,
    ], dtype=float),
    "yolo11x_bs48": np.array([
        315.63, 141.7, 324.83, 101.94, 194.74, 242.18, 246.62, 259.38, 259.16,
        257.99, 256.27, 294.88, 343.27, 308.64, 518.13, 660.36, 420.49, 449.52,
        499.88, 472.26, 486.56, 496.15, 484.8,
    ], dtype=float),
    "yolo11x_bs64": np.array([
        413.0, 188.86, 432.53, 135.87, 259.57, 322.7, 328.44, 345.43, 345.24,
        343.61, 341.33, 392.93, 456.53, 409.03, 679.79, 878.63, 559.64, 598.38,
        665.62, 629.49, 648.56, 661.4, 646.11,
    ], dtype=float),
}

# ── constants ─────────────────────────────────────────────────────────────────
N_YOLO_LAYERS = 24

FEEDBACK_FILE   = "cpcdrl_feedback.json"    # legacy — prefer feedback_file_path(device)
CONTROLLER_SAVE = "cpcdrl_controller.pt"    # legacy — prefer controller_save_path(device)

# DEVICE_CATALOGUE is loaded from devices.yaml above via _load_devices()


def normalize_device_name(name: str) -> str:
    """Normalize device name: 'DEVICE 2' → 'DEVICE_2', case-insensitive."""
    if name is None:
        return "UNKNOWN"
    return str(name).strip().upper().replace(" ", "_")


def controller_save_path(device_name: str) -> str:
    return f"cpcdrl_controller_{normalize_device_name(device_name)}.pt"


def feedback_file_path(device_name: str) -> str:
    return f"cpcdrl_feedback_{normalize_device_name(device_name)}.json"


def get_device_timing(device_name: str) -> np.ndarray:
    """Per-layer timing (seconds) for a device — direct measurement, no scaling."""
    device_name = normalize_device_name(device_name)
    timing = _TIMING_BY_DEVICE.get(device_name)
    if timing is None:
        raise ValueError(f"Unknown device '{device_name}'. Valid: {list(DEVICE_CATALOGUE)}")
    return timing.copy()


def get_device_bandwidth_mbps(edge_device: str) -> float:
    """Link bandwidth from edge device to DAI, in MB/s."""
    edge_device = normalize_device_name(edge_device)
    spec        = DEVICE_CATALOGUE.get(edge_device, {})
    link_gbps   = spec.get("link_gbps")
    if link_gbps is None:
        return 115.0
    return link_gbps * 1000.0 / 8.0


def slowest_device(device_names) -> str:
    """Return the device name with lowest GFLOPS (most conservative for cut decision)."""
    if isinstance(device_names, str):
        return normalize_device_name(device_names)
    names = [normalize_device_name(n) for n in device_names]
    return min(names, key=lambda n: DEVICE_CATALOGUE.get(n, {}).get("gflops", 1e9))


# ── CPCDRLProfile builder ─────────────────────────────────────────────────────

def make_real_profile(
    edge_device:    str   = "DEVICE_1",
    model_key:      str   = "yolo26n_bs32",
    bandwidth_mbps: float = None,
):
    """
    Build a CPCDRLProfile from embedded per-layer timing and cut sizes.

    fe_mflops = fc_mflops = 1e-6 so that compute_latency() returns seconds
    directly (used by DDPG._build_layer_state() for state normalisation only).
    Actual latency is computed by compute_latency_real().
    """
    from cpcdrl_baseline import CPCDRLProfile

    edge_timing = get_device_timing(edge_device)
    if bandwidth_mbps is None:
        bandwidth_mbps = get_device_bandwidth_mbps(edge_device)

    data_sizes = CUT_DATA_SIZES_MB_BY_MODEL.get(model_key)
    if data_sizes is not None:
        layer_output_mb = np.append(data_sizes[:N_YOLO_LAYERS - 1], 0.0)
    else:
        layer_output_mb = np.full(N_YOLO_LAYERS, 50.0)

    return CPCDRLProfile(
        layer_macs      = edge_timing,
        layer_output_mb = layer_output_mb,
        fe_mflops       = 1e-6,
        fc_mflops       = 1e-6,
        bandwidth_kbps  = bandwidth_mbps * 8.0 * 1000.0,
    )


# ── latency estimator ─────────────────────────────────────────────────────────

def compute_latency_real(
    cut_point:          int,
    compression_ratios: np.ndarray,
    edge_timing:        np.ndarray,
    model_key:          str,
    bandwidth_mbps:     float,
    cloud_timing:       np.ndarray = None,
) -> float:
    """
    Estimate E2E latency (seconds) using embedded per-layer timing.
    cloud_timing defaults to DAI measured timing if not supplied.
    """
    if cloud_timing is None:
        cloud_timing = _TIMING_BY_DEVICE["DAI"]

    if cut_point <= 0:
        t_edge = t_trans = 0.0
    else:
        t_edge = float(np.sum(compression_ratios[:cut_point] * edge_timing[:cut_point]))

        data_sizes  = CUT_DATA_SIZES_MB_BY_MODEL.get(model_key)
        size_8bit   = float(data_sizes[min(cut_point - 1, len(data_sizes) - 1)]) \
                      if data_sizes is not None else 50.0
        mean_ratio  = float(np.mean(compression_ratios[:cut_point]))
        t_trans     = size_8bit * mean_ratio / max(bandwidth_mbps, 0.1)

    t_cloud = float(np.sum(cloud_timing[cut_point:]))
    return t_edge + t_trans + t_cloud


# ── per-layer ratio → num_bits ────────────────────────────────────────────────

def ratios_to_bits_per_layer(compression_ratios: np.ndarray, cut_point: int) -> list:
    """Map DDPG per-layer ratios to quantisation bit-depths (paper-faithful)."""
    if cut_point <= 0:
        return []
    bits = []
    for i in range(cut_point):
        r = float(compression_ratios[i])
        bits.append(max(4, min(16, int(round(4.0 + r * 12.0)))))
    return bits


def mean_ratio_to_num_bits(compression_ratios: np.ndarray, cut_point: int) -> int:
    """Mean of per-layer bits — scalar summary for feedback logging."""
    bits = ratios_to_bits_per_layer(compression_ratios, cut_point)
    return int(round(sum(bits) / len(bits))) if bits else 8


# ── model key resolution ──────────────────────────────────────────────────────

def resolve_model_key(model_name: str, batch_size: int) -> str:
    for candidate in [f"{model_name}_bs{batch_size}", model_name]:
        if candidate in CUT_DATA_SIZES_MB_BY_MODEL:
            return candidate
    for key in CUT_DATA_SIZES_MB_BY_MODEL:
        if model_name in key:
            return key
    keys = list(CUT_DATA_SIZES_MB_BY_MODEL.keys())
    return keys[0] if keys else ""


# ── session feedback I/O ──────────────────────────────────────────────────────

def write_feedback(device_name: str, cut_int: int, num_bits: int,
                   e2e_latency_ms: float, map50: float) -> None:
    with open(feedback_file_path(device_name), "w") as f:
        json.dump({
            "cut_int":        int(cut_int),
            "num_bits":       int(num_bits),
            "e2e_latency_ms": float(e2e_latency_ms),
            "map50":          float(map50),
        }, f)


def read_feedback(device_name: str) -> Optional[dict]:
    path = feedback_file_path(device_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
