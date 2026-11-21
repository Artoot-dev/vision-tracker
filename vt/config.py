"""Default configuration. Every knob the pipeline reads lives here.

Two named profiles ship in config.yaml:
  balanced  everyday tracking, permissive template updates
  strict    the spec's literal gate (update only when detection confidence > 0.95)
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "profile": "balanced",
    "video": {
        "stride": 1,
        "start_frame": 0,
        "max_frames": 0,
        "fps_override": 0.0,
    },
    "camera": {
        "hfov_deg": 65.0,
        "focal_px": 0.0,        # set explicitly if you calibrated; else derived from hfov
        "ipd_m": 0.063,
    },
    "depth": {
        "enable": True,
        "mode": "ipd",          # ipd | pseudo | off
        "person_height_m": 1.70,
    },
    "face": {
        "enable": True,
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "every_n": 1,
    },
    "detector": {
        "backend": "auto",      # auto | ultralytics | onnx | dummy
        "model": "yolov8n.pt",
        "imgsz": 416,
        "conf": 0.25,
        "iou": 0.5,
        "classes": [0],
        "device": "",
        "half": False,
        "max_det": 50,
        "onnx_format": "auto",
        "normalize": True,
        "verbose": False,
        "dummy_gt": None,
        "redetect_conf": 0.15,
        "every_n": 1,           # run YOLO every Nth frame; BoT-SORT carries the rest
    },
    "tracker": {
        "name": "botsort",              # botsort | bytetrack | none (own association)
        "config": "configs/botsort_reid.yaml",
        "process_noise": 40.0,
        "meas_noise": 3.0,
        "coast_max_frames": 30,
        "lost_max_frames": 240,
        "redetect_every": 5,
        "uncertainty_growth": 1.12,
        "coast_size_growth": 0.02,
        "search_base_px": 28.0,
        "search_growth": 0.25,
        "search_diag_frac": 0.35,
        "iou_min": 0.20,
        "iou_min_coast": 0.02,
        "iou_strong": 0.50,
        "sim_min": 0.50,
        "sim_min_lock": 0.40,
        "hard_sim_floor": 0.30,
        "maha_gate": 4.0,
        "w_iou": 1.0,
        "w_emb": 2.0,
        "w_motion": 0.5,
        "w_id": 0.8,
        "size_alpha": 0.25,
        "min_area_px": 16.0,
        "trail_len": 180,
        "gmc": True,
        "flow_refine": False,
        "own_ids": True,
    },
    "reid": {
        "embedder": "lite",             # lite | mobilenet | none
        "weights": None,
        "size": [32, 64],
        "hist_h": 8,
        "hist_s": 8,
        "mask": "ellipse",
        "crop_pad": 0.0,
        "sat_min": 25,
        "block_weights": [2.0, 1.0, 1.0, 1.0],
        "ema_alpha": 0.10,
        "update_score_gate": 0.50,      # strict profile: 0.95
        "update_sim_min": 0.55,
        "drift_floor": 0.40,
        "gallery_size": 8,
        "gallery_dedupe": 0.98,
    },
    "sync": {
        "motor_hz": 1000.0,
        "latency_comp": True,
        "max_extrapolate_ms": 120.0,
        "fixed_latency_ms": 0.0,
    },
    "live": {
        "window": "vision tracker",
        "window_enabled": True,
        "speed": 1.0,               # 1.0 = real time, 0 = as fast as possible
        "catch_up": True,           # drop late frames so the view stays real-time
        "max_skip": 30,
        "progress_every": 60,
        "reuse_selection_frame": True,
        "quiet": False,
    },
    "output": {
        "dir": "out",
        "video": None,
        "jsonl": None,
        "stats": None,
        "codec": "mp4v",
        "h264": True,
        "draw_trail": 180,
        "hud": True,
        "points": False,            # draw the trail as tracking points
        "keypoints": False,         # draw pose keypoints when the model provides them
        "all_tracks": True,         # draw every detection, or only the subject
        "save_selection_png": True,
        "progress_every": 25,
    },
    "selection": {
        "mode": "click",                # click | auto | point | roi
        "at_sec": 0.0,
        "auto_rule": "largest",         # largest | conf
        "point": None,
        "roi": None,
        "scan_sec": 2.0,
    },
}

PROFILES = {
    "balanced": {},
    "strict": {
        "reid": {"update_score_gate": 0.95, "ema_alpha": 0.05, "update_sim_min": 0.70,
                 "drift_floor": 0.55},
    },
}


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path).expanduser()
        if p.exists():
            with open(p) as fh:
                user = yaml.safe_load(fh) or {}
            profile = user.get("profile")
            if profile and profile in PROFILES:
                cfg = _deep_merge(cfg, PROFILES[profile])
            cfg = _deep_merge(cfg, {k: v for k, v in user.items() if k != "profiles"})
    else:
        p = Path(__file__).resolve().parent.parent / "config.yaml"
        if p.exists():
            with open(p) as fh:
                user = yaml.safe_load(fh) or {}
            profile = user.get("profile")
            if profile and profile in PROFILES:
                cfg = _deep_merge(cfg, PROFILES[profile])
            cfg = _deep_merge(cfg, {k: v for k, v in user.items() if k != "profiles"})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return cfg
