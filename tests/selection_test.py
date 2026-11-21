#!/usr/bin/env python3
"""Selection-mode tests: the headless paths the CLI exposes.

Verifies that --select auto / point / roi and --print-config resolve to the
intended detection, since the GUI click path cannot be driven from a test.

  python tests/selection_test.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vt.config import load_config  # noqa: E402
from vt.detect import Detection  # noqa: E402
from vt.select import pick_headless, resolve_selection  # noqa: E402

# a small box, a large box containing the small one, and a medium box elsewhere
DETS = [
    Detection(xyxy=np.array([100, 100, 300, 400], dtype=np.float64), score=0.4,
              cls_id=0, cls_name="person"),
    Detection(xyxy=np.array([150, 150, 200, 250], dtype=np.float64), score=0.9,
              cls_id=0, cls_name="person"),
    Detection(xyxy=np.array([400, 50, 480, 200], dtype=np.float64), score=0.7,
              cls_id=0, cls_name="person"),
]


def test_pick_modes():
    assert pick_headless(DETS, mode="auto", rule="largest") == 0
    assert pick_headless(DETS, mode="auto", rule="conf") == 1
    # a point inside the nested small box must select the SMALLEST containing box,
    # otherwise a nested target is unreachable
    assert pick_headless(DETS, mode="point", point=(175, 200)) == 1
    assert pick_headless(DETS, mode="point", point=(120, 380)) == 0
    assert pick_headless(DETS, mode="point", point=(600, 400)) is None
    assert pick_headless(DETS, mode="roi", roi=(395, 45, 90, 160)) == 2
    assert pick_headless([], mode="auto") is None
    print("selection: pick_headless modes OK")


def test_resolve_and_headless_pipeline():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cfg = load_config(overrides={"selection": {"mode": "point", "point": [175, 200]}})
    idx, detail = resolve_selection(cfg["selection"], frame, DETS)
    assert idx == 1, (idx, detail)
    assert detail["bbox"] == [150.0, 150.0, 200.0, 250.0], detail
    cfg = load_config(overrides={"selection": {"mode": "auto", "auto_rule": "conf"}})
    idx, _ = resolve_selection(cfg["selection"], frame, DETS)
    assert idx == 1
    idx, detail = resolve_selection(cfg["selection"], frame, [])
    assert idx is None and "no detections" in detail["reason"]
    print("selection: resolve_selection OK")


def test_cli_smoke():
    """Drive run.py the way the operator will: real argparse, real config merge."""
    out = Path(tempfile.mkdtemp(prefix="vt_cli_check_"))
    gt = ROOT / "out" / "synthetic_gt.json"
    if not gt.exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"),
                        "--out", str(ROOT / "out" / "synthetic.mp4"), "--gt", str(gt)],
                       check=True, capture_output=True)
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), str(ROOT / "out" / "synthetic.mp4"),
                        "--backend", "dummy", "--dummy-gt", str(gt),
                        "--select", "auto", "--select-at", "0.5", "--no-face",
                        "--depth", "off", "--no-h264", "--max-frames", "40",
                        "--out-dir", str(out)],
                       capture_output=True, text=True)
    print(r.stdout[-1200:] if r.stdout else r.stderr[-2000:])
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "synthetic_tracked.mp4").exists()
    assert (out / "synthetic_track.jsonl").exists()

    c = subprocess.run([sys.executable, str(ROOT / "run.py"), "--print-config"],
                       capture_output=True, text=True)
    assert c.returncode == 0 and '"profile"' in c.stdout, c.stderr[-800:]
    # a missing model must fail with guidance, not an onnxruntime stack trace
    bad = subprocess.run([sys.executable, str(ROOT / "run.py"), str(ROOT / "out" / "synthetic.mp4"),
                          "--backend", "onnx", "--model", "definitely_missing.onnx",
                          "--select", "auto", "--max-frames", "1"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "not found" in (bad.stdout + bad.stderr), \
        (bad.returncode, bad.stdout[-400:], bad.stderr[-400:])
    print("selection: run.py CLI OK")


if __name__ == "__main__":
    test_pick_modes()
    test_resolve_and_headless_pipeline()
    test_cli_smoke()
