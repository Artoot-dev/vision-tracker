#!/usr/bin/env python3
"""Live-mode tests.

Exercises the live runner headlessly (no window, no display needed) and checks
the two things that make it useful:

  1. it paces a file to real time, so what you see is what a live camera feeds
  2. the fallback accounting reflects what actually happened, including the case
     where the detector only runs every Nth frame

  python tests/live_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vt.config import load_config  # noqa: E402
from vt.live import LiveRunner  # noqa: E402

HIDE_FROM, HIDE_TO = 35, 70


def _make_clip(tmp: Path):
    video = tmp / "synthetic.mp4"
    gt = tmp / "synthetic_gt.json"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"),
                    "--out", str(video), "--gt", str(gt),
                    "--occlude-from", str(HIDE_FROM), "--occlude-to", str(HIDE_TO)],
                   check=True, capture_output=True)
    return video, gt


def _cfg(tmp, gt, **over):
    base = {
        "detector": {"backend": "dummy", "dummy_gt": str(gt), "classes": [0]},
        "selection": {"mode": "auto", "auto_rule": "largest", "at_sec": 0.0},
        "output": {"dir": str(tmp / "out"), "h264": False, "stats": None},
        "live": {"window_enabled": False, "speed": 0.0, "quiet": True,
                 "progress_every": 1000},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return load_config(overrides=base)


def test_live_headless_and_accounting():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video, gt = _make_clip(tmp)
        cfg = _cfg(tmp, gt)
        summary = LiveRunner(cfg, video).run()
        assert summary["ok"], summary
        fb = summary["fallback"]
        print("\nlive: fallback", json.dumps({k: fb[k] for k in
              ("frames", "sustained_frames", "fallback_frames", "coast_frames",
               "lost_frames", "max_gap_frames", "yolo_redetect_calls",
               "reacquisitions", "id_switches", "new_tracks_after_first_frame")}, indent=2))

        assert summary["frames_processed"] >= 90, summary["frames_processed"]
        # the subject is hidden for 36 frames, longer than the 30-frame coast budget
        assert fb["fallback_frames"] >= 30, fb
        assert fb["coast_frames"] >= 25, fb
        assert fb["lost_frames"] >= 1, fb
        assert fb["max_gap_frames"] >= 30, fb
        # the pipeline literally called YOLO again while LOST
        assert fb["yolo_redetect_calls"] >= 1, fb
        # and it got the subject back
        assert fb["reacquisitions"] >= 1, fb
        assert fb["sustained_frames"] > 50, fb
        # gaps must line up with the hidden window
        gaps = [g for g in fb["gaps"] if g["start_frame"] >= HIDE_FROM - 2]
        assert gaps, fb["gaps"]
        assert gaps[0]["start_frame"] <= HIDE_TO, gaps[0]
        sp = Path(summary["stats_path"])
        assert sp.exists() and json.load(open(sp))["fallback"]["fallback_frames"] == fb["fallback_frames"]
        print("live: PASS (headless real-time pacing + fallback accounting)")


def test_pacing_is_real_time():
    """speed 1.0 on a 96-frame 30 fps clip must take about 3.2 s of wall clock."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video, gt = _make_clip(tmp)
        cfg = _cfg(tmp, gt, live={"speed": 1.0, "catch_up": True})
        t0 = time.perf_counter()
        summary = LiveRunner(cfg, video).run()
        wall = time.perf_counter() - t0
        expected = summary["frames_processed"] / 30.0
        print(f"\nlive: paced {summary['frames_processed']} frames in {wall:.2f}s "
              f"(real time would be {expected:.2f}s, dropped {summary['dropped_frames']})")
        assert summary["ok"]
        # allow generous slack for a slow detector on a shared CPU
        assert wall >= expected * 0.7, (wall, expected)
        assert wall <= expected + 6.0, (wall, expected)
        print("live: PASS (real-time pacing)")


def test_detect_every_n():
    """With the detector running every 3rd frame the tracker must carry the rest,
    and the accounting must show the extra fallback frames."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video, gt = _make_clip(tmp)
        cfg = _cfg(tmp, gt, detector={"backend": "dummy", "dummy_gt": str(gt),
                                      "classes": [0], "every_n": 3})
        summary = LiveRunner(cfg, video).run()
        fb = summary["fallback"]
        print(f"\nlive: detect every 3rd frame -> confirmed {fb['sustained_frames']}, "
              f"fallback {fb['fallback_frames']}, yolo calls {summary['frames_processed'] // 3}")
        assert summary["detect_every"] == 3, summary["detect_every"]
        # frames where the detector did not run cannot be detection-confirmed
        assert fb["sustained_frames"] <= summary["frames_processed"] * 0.7, fb
        assert fb["fallback_frames"] >= summary["frames_processed"] * 0.3, fb
        print("live: PASS (detect-every-N accounting)")


if __name__ == "__main__":
    test_live_headless_and_accounting()
    test_pacing_is_real_time()
    test_detect_every_n()
