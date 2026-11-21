#!/usr/bin/env python3
"""Generate a synthetic clip + ground-truth boxes for smoke testing.

Scene, 96 frames at 30 fps:
  target      teal figure, walks right, parks fully behind a gray pillar for
              frames 35-70 (36 frames, longer than the 30-frame coast budget so
              the lock must fall through COASTING into LOST), then walks out
  distractor  red figure in a separate vertical band, always visible, so the
              appearance gate has to reject it while the target is hidden
  pillar      opaque, covers the target's whole band

The ground-truth file omits the target while it is hidden, which is what a real
detector does. That makes coast -> LOST -> re-ID recovery testable with no model.

  python tools/make_test_video.py --out out/synthetic.mp4 --gt out/synthetic_gt.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"opencv required: {exc}")

PILLAR_X = (250, 395)
PILLAR_Y = (215, 480)


def draw_person(img, x, y, h, color, accent, phase=0.0):
    """Crude person: torso, head, arms. Distinct internal pattern so the appearance
    descriptor has something to lock onto."""
    h = max(int(round(h)), 4)
    w = int(h * 0.42)
    x, y = int(round(x)), int(round(y))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x + int(w * 0.15), y + int(h * 0.15)),
                  (x + int(w * 0.85), y + int(h * 0.45)), accent, -1)
    head_r = max(int(h * 0.13), 3)
    cv2.circle(img, (x + w // 2, y - head_r), head_r, color, -1)
    cv2.circle(img, (x + w // 2, y - head_r), max(head_r - 3, 1), (235, 235, 235), -1)
    arm = int(h * 0.06) + int(6 * np.sin(phase))
    cv2.rectangle(img, (x - arm, y + int(h * 0.2)), (x + 2, y + int(h * 0.28)), accent, -1)
    cv2.rectangle(img, (x + w - 2, y + int(h * 0.2)), (x + w + arm, y + int(h * 0.28)),
                  accent, -1)
    return np.array([x - arm, y - head_r * 2, x + w + arm, y + h], dtype=np.float64)


def target_x(i, hide_from, hide_to):
    """Deterministic path: walk in, park behind the pillar, walk out."""
    if i < hide_from:
        return 118.0 + (i / max(hide_from, 1)) * 130.0
    if i <= hide_to:
        return 300.0  # spans 300..372 with shoulders: inside the pillar (250..395)
    return 405.0 + min((i - hide_to - 1) * 4.0, 95.0)


def target_y(i, height, frame_h):
    return frame_h - height - 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/synthetic.mp4")
    ap.add_argument("--gt", default="out/synthetic_gt.json")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--occlude-from", type=int, default=35)
    ap.add_argument("--occlude-to", type=int, default=70)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bg = rng.integers(40, 90, size=(a.h, a.w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    for _ in range(60):
        p = (int(rng.integers(0, a.w)), int(rng.integers(0, a.h)))
        cv2.circle(bg, p, int(rng.integers(4, 26)),
                   tuple(int(v) for v in rng.integers(60, 160, 3)), -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.2)

    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.w, a.h))
    gt = {"frames": {}}
    expected_center = {}
    for i in range(a.frames):
        img = bg.copy()
        th = 150 + 22 * np.sin(2 * np.pi * i / 48.0)
        tx = target_x(i, a.occlude_from, a.occlude_to)
        ty = target_y(i, th, a.h)
        tbox = draw_person(img, tx, ty, th, (170, 165, 60), (60, 200, 255), phase=i * 0.3)

        # distractor: separate vertical band (top of frame), always visible
        dx = 560 - 420 * (i / max(a.frames - 1, 1))
        dbox = draw_person(img, dx, 60, 118, (40, 40, 205), (200, 220, 220), phase=-i * 0.4)

        # opaque pillar over the target's band only
        cv2.rectangle(img, (PILLAR_X[0], PILLAR_Y[0]), (PILLAR_X[1], PILLAR_Y[1]),
                      (105, 105, 105), -1)
        for k in range(PILLAR_X[0], PILLAR_X[1], 26):
            cv2.line(img, (k, PILLAR_Y[0]), (k + 13, PILLAR_Y[1]), (125, 125, 125), 2)

        img = np.clip(img.astype(np.int16) + rng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
        writer.write(img)

        hidden = a.occlude_from <= i <= a.occlude_to
        entries = []
        if not hidden:
            entries.append([float(v) for v in tbox])
            expected_center[str(i)] = [float((tbox[0] + tbox[2]) / 2), float((tbox[1] + tbox[3]) / 2)]
        entries.append([float(v) for v in dbox])
        gt["frames"][str(i)] = entries
    writer.release()
    gt["meta"] = {
        "target_index_in_frame": 0,
        "hidden_frames": [a.occlude_from, a.occlude_to],
        "distractor_index_in_frame": -1,
        "pillar": {"x": list(PILLAR_X), "y": list(PILLAR_Y)},
        "expected_target_center": expected_center,
        "n_frames": a.frames, "fps": a.fps, "size": [a.w, a.h],
    }
    Path(a.gt).parent.mkdir(parents=True, exist_ok=True)
    with open(a.gt, "w") as fh:
        json.dump(gt, fh)
    print(f"wrote {a.out} ({a.frames} frames) and {a.gt}")


if __name__ == "__main__":
    main()
