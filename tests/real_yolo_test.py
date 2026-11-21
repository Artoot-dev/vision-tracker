#!/usr/bin/env python3
"""Real-detector end-to-end test: YOLO + BoT-SORT + the subject lock.

Unlike tests/smoke_test.py (no model, ground-truth boxes), this one runs the real
ultralytics backend on a real image with real people, with a panning camera to
exercise global motion, and an occluder that crosses the chosen subject so the
coast -> LOST -> re-ID path is exercised on genuine detections.

Needs the optional backend:
    pip install "ultralytics>=8.2,<9"
It is skipped with a clear message when ultralytics is not importable.

  python tests/real_yolo_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vt.config import load_config  # noqa: E402
from vt.geometry import iou_xyxy  # noqa: E402
from vt.pipeline import Pipeline  # noqa: E402

SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"
HIDE_FROM, HIDE_TO = 36, 66
N_FRAMES = 96
CANVAS = (960, 720)


def fetch_sample(dest: Path):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SAMPLE_URL}")
    urllib.request.urlretrieve(SAMPLE_URL, dest)
    return dest


def build_clip(img_path: Path, video_path: Path, gt_path: Path):
    import cv2
    from ultralytics import YOLO

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"could not read {img_path}")
    h, w = img.shape[:2]
    cw, ch = min(CANVAS[0], w), min(CANVAS[1], h)

    # pick the subject on the full image the way the operator would: largest person
    model = YOLO("yolov8n.pt")
    r = model.predict(img, imgsz=416, conf=0.3, classes=[0], verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise SystemExit("no person detected in the sample frame")
    boxes = r.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    subject = boxes[int(np.argmax(areas))].astype(np.float64)  # in IMAGE coords
    print(f"subject in image coords: {[round(float(v)) for v in subject]} "
          f"(area {int(areas.max())} px)")

    def feasible(lo, hi, size, total):
        """Window-origin range that keeps [lo, hi] inside the window at all times."""
        a = max(0.0, hi - size)
        b = min(float(total - size), lo)
        if b < a:
            a = b = max(0.0, min(float(total - size), (lo + hi) / 2.0 - size / 2.0))
        return a, b

    xa, xb = feasible(subject[0], subject[2], cw, w)
    ya, yb = feasible(subject[1], subject[3], ch, h)
    print(f"pan window origin: x {xa:.0f}..{xb:.0f}  y {ya:.0f}..{yb:.0f}")

    frames = []
    for i in range(N_FRAMES):
        u = i / max(N_FRAMES - 1, 1)
        # bounded jitter keeps the camera moving even when the feasible range is
        # a single point (a static camera would make the test weaker, not easier)
        jx = 10.0 * np.sin(2 * np.pi * u * 2.0)
        jy = 7.0 * np.sin(2 * np.pi * u * 3.0)
        x0 = int(round(min(max(xa + (xb - xa) * u + jx, 0.0), w - cw)))
        y0 = int(round(min(max(ya + (yb - ya) * u + jy, 0.0), h - ch)))
        frames.append({"img": img[y0:y0 + ch, x0:x0 + cw].copy(), "off": (x0, y0)})

    for i, f in enumerate(frames):
        if HIDE_FROM <= i <= HIDE_TO:
            x0, y0 = f["off"]
            x1 = max(int(subject[0] - x0) - 26, 0)
            x2 = min(int(subject[2] - x0) + 26, cw)
            y1 = max(int(subject[1] - y0) - 26, 0)
            y2 = min(int(subject[3] - y0) + 26, ch)
            f["img"][y1:y2, x1:x2] = (108, 108, 108)
            for k in range(x1, x2, 24):
                cv2.line(f["img"], (k, y1), (k + 12, y2), (128, 128, 128), 2)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cw, ch))
    for f in frames:
        writer.write(f["img"])
    writer.release()
    with open(gt_path, "w") as fh:
        json.dump({"subject_image_coords": [float(v) for v in subject],
                   "hidden_frames": [HIDE_FROM, HIDE_TO], "n_frames": N_FRAMES}, fh)
    print(f"wrote {video_path} ({N_FRAMES} frames, {cw}x{ch}) and {gt_path}")
    return subject, (cw, ch)


def main():
    try:
        import ultralytics  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        print(f"SKIP: ultralytics/torch not available ({type(exc).__name__}: {exc})")
        print('      pip install "ultralytics>=8.2,<9"')
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        img = fetch_sample(tmp / "bus.jpg")
        video = tmp / "real.mp4"
        gt_path = tmp / "real_gt.json"
        subject, (cw, ch) = build_clip(img, video, gt_path)

        cfg = load_config(overrides={
            "detector": {"backend": "ultralytics", "model": "yolov8n.pt", "imgsz": 416,
                         "conf": 0.25, "classes": [0], "device": "cpu"},
            "tracker": {"name": "botsort", "coast_max_frames": 8, "lost_max_frames": 240},
            "selection": {"mode": "auto", "auto_rule": "largest", "at_sec": 0.0},
            "depth": {"mode": "off"},
            "face": {"enable": False},
            "output": {"dir": str(tmp / "out"), "h264": False, "progress_every": 20},
        })
        summary = Pipeline(cfg, video).run()
        assert summary["ok"], summary
        recs = [json.loads(l) for l in open(summary["outputs"]["jsonl"])]
        states = [r["state"] for r in recs]
        counts = {s: states.count(s) for s in sorted(set(states))}
        print("\nreal: states", counts)
        print("real: tracker", summary["tracker_stats"])
        print(f"real: avg det {summary['avg_det_ms']}ms  avg loop {summary['avg_loop_ms']}ms  "
              f"fps {summary['end_to_end_fps']}")

        assert summary["tracker_stats"]["states"]["LOCKED"] > 40, counts
        # the real detector does find people; the lock must stay on the chosen one
        locked = [r for r in recs if r["state"] == "LOCKED"]
        assert locked, "never locked"
        # the lock must sit ON a real YOLO detection, not float on its own
        ious = []
        for r in locked:
            if r.get("detections"):
                ious.append(max(iou_xyxy(r["bbox"], d[:4]) for d in r["detections"]))
        assert ious, "no detection overlapped the locked box"
        assert min(ious) > 0.35, f"lock drifted off detections: min IoU {min(ious):.2f}"
        # and it must follow the subject rather than sit on a static box
        xs = [r["center"][0] for r in locked]
        ys = [r["center"][1] for r in locked]
        assert max(max(xs) - min(xs), max(ys) - min(ys)) > 12, \
            f"tracked centre barely moved: x {min(xs):.0f}..{max(xs):.0f} y {min(ys):.0f}..{max(ys):.0f}"
        # during the occluder the tracker must not accept a different person
        hidden = [r for r in recs if HIDE_FROM + 2 <= r["frame_index"] <= HIDE_TO - 2]
        assert hidden, "no frames inside the occluder window"
        assert not all(r["measured"] for r in hidden), \
            "every occluded frame was matched: the occluder is not actually hiding the subject"
        fb = summary["fallback"]
        # the gap is longer than the coast budget, so the tracker must have fallen
        # through to LOST and asked YOLO again
        assert fb["coast_frames"] >= 2, fb
        assert fb["lost_frames"] >= 1, fb
        assert fb["yolo_redetect_calls"] >= 1, fb
        assert fb["reacquisitions"] >= 1, fb
        assert summary["tracker_stats"]["relinks"] >= 1, summary["tracker_stats"]
        # ...and it must end up locked on, not abandoned
        tail = recs[-10:]
        assert all(r["state"] == "LOCKED" for r in tail), [r["state"] for r in tail]

        # ---- live mode on the same real footage, real YOLO, real BoT-SORT ----
        from vt.live import LiveRunner
        live_cfg = load_config(overrides={
            "detector": {"backend": "ultralytics", "model": "yolov8n.pt", "imgsz": 416,
                         "conf": 0.25, "classes": [0], "device": "cpu"},
            "tracker": {"name": "botsort", "coast_max_frames": 8},
            "selection": {"mode": "auto", "auto_rule": "largest"},
            "depth": {"mode": "off"},
            "face": {"enable": False},
            "output": {"dir": str(tmp / "out_live"), "h264": False},
            "live": {"window_enabled": False, "speed": 0.0, "quiet": True,
                     "progress_every": 1000},
            "video": {"max_frames": HIDE_TO + 6},
        })
        lsum = LiveRunner(live_cfg, video).run()
        lfb = lsum["fallback"]
        print("\nreal-live: fallback", {k: lfb[k] for k in
              ("frames", "sustained_frames", "fallback_frames", "coast_frames",
               "lost_frames", "max_gap_frames", "yolo_redetect_calls",
               "reacquisitions", "id_switches")})
        assert lsum["ok"], lsum
        assert lfb["sustained_frames"] > 20, lfb
        assert lfb["coast_frames"] >= 1, lfb
        assert lfb["lost_frames"] >= 1, lfb
        assert lfb["yolo_redetect_calls"] >= 1, lfb
        assert lfb["reacquisitions"] >= 1, lfb
        assert lsum["final_state"] == "LOCKED", lsum.get("final_state")
        print(f"real-live: {lsum['frames_processed']} frames, "
              f"avg detect {lsum['avg_detect_ms']}ms, "
              f"{'LIVE' if lsum['mode'] == 'playback' else lsum['mode']}")
        print("real: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
