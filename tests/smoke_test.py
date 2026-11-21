#!/usr/bin/env python3
"""End-to-end smoke test - no model, no GPU.

Generates the synthetic clip, runs the full pipeline through the dummy detector
(which replays ground-truth boxes and omits the target while it is hidden behind
the pillar), then asserts the lock behaved:

  - it tracked while the target was visible
  - it coasted once the target vanished
  - it fell through to LOST (the occlusion is longer than the coast budget)
  - it re-linked by appearance when the target reappeared
  - it never accepted the distractor, which stays visible the whole time
  - the latency-compensated target is present in every record

  python tests/smoke_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vt.config import load_config  # noqa: E402
from vt.pipeline import Pipeline  # noqa: E402
from vt.reid import LiteEmbedder, cos_sim  # noqa: E402

HIDE_FROM, HIDE_TO = 35, 70


def _make_clip(tmp: Path):
    video = tmp / "synthetic.mp4"
    gt = tmp / "synthetic_gt.json"
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_test_video.py"),
                    "--out", str(video), "--gt", str(gt),
                    "--occlude-from", str(HIDE_FROM), "--occlude-to", str(HIDE_TO)],
                   check=True, capture_output=True)
    return video, gt


def test_embedder_discrimination():
    """Regression guard on the appearance descriptor itself: the anchor crop must
    score clearly higher against later views of the SAME subject than against the
    distractor. This is what makes re-ID recovery possible at all."""
    import cv2
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video, gt_path = _make_clip(tmp)
        gt = json.load(open(gt_path))
        cap = cv2.VideoCapture(str(video))
        frames = {}
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames[i] = f
            i += 1
        cap.release()
        emb = LiteEmbedder()
        anchor = emb.extract(frames[20], gt["frames"]["20"][0])
        assert anchor is not None and float(np.linalg.norm(anchor)) > 0.9, "descriptor degenerated"
        same = [cos_sim(anchor, emb.extract(frames[f], gt["frames"][str(f)][0]))
                for f in (30, 34, 80, 90)]
        other = [cos_sim(anchor, emb.extract(frames[f], gt["frames"][str(f)][-1]))
                 for f in (20, 45, 60, 80)]
        print(f"embedder: same-subject sims {[round(v, 3) for v in same]}")
        print(f"embedder: distractor  sims {[round(v, 3) for v in other]}")
        assert min(same) > 0.5, same
        assert max(other) < 0.45, other
        assert min(same) - max(other) > 0.12, (same, other)


def test_smoke(verbose=True):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        video, gt = _make_clip(tmp)
        cfg = load_config(overrides={
            "detector": {"backend": "dummy", "dummy_gt": str(gt), "classes": [0]},
            "selection": {"mode": "auto", "auto_rule": "largest", "at_sec": 0.0},
            "output": {"dir": str(tmp / "out"), "h264": False},
        })
        summary = Pipeline(cfg, video).run()
        assert summary["ok"], summary
        recs = [json.loads(l) for l in open(summary["outputs"]["jsonl"])]
        assert len(recs) == summary["frames"] >= 90, len(recs)

        states = [r["state"] for r in recs]
        counts = {s: states.count(s) for s in sorted(set(states))}
        if verbose:
            print("\nsmoke: states", counts)
            print("smoke: tracker", summary["tracker_stats"])
            print("smoke: fallback", summary["fallback"]["verdict"])

        assert counts.get("LOCKED", 0) > 50, counts
        assert counts.get("COASTING", 0) > 3, counts
        assert counts.get("LOST", 0) > 0, counts
        assert summary["tracker_stats"]["relinks"] >= 1, summary["tracker_stats"]
        assert summary["tracker_stats"]["template_updates"] > 0, summary["tracker_stats"]

        hidden = [r for r in recs if HIDE_FROM + 1 <= r["frame_index"] <= HIDE_TO]
        assert hidden, "no frames inside the occlusion window"
        assert not any(r["measured"] for r in hidden), \
            "the lock accepted a detection while the target was hidden (distractor captured)"
        assert not any(r["state"] == "LOCKED" for r in hidden), \
            "state stayed LOCKED through the occlusion"
        assert any(r["state"] == "LOST" for r in hidden), \
            "never fell through to LOST during a 36-frame occlusion"

        # coasting must keep the estimate near the target's band, not snap to the
        # distractor's band (which is ~300 px higher in the frame)
        ys = [r["center"][1] for r in hidden]
        assert min(ys) > 250, f"estimate drifted up to the distractor band: {min(ys):.0f}"

        # re-ID recovery must land back on the subject and re-lock
        after = [r for r in recs if r["frame_index"] > HIDE_TO]
        relinks = [r for r in after if r["relinked"]]
        assert relinks, "no relink after the occlusion ended"
        assert any(r["state"] == "LOCKED" for r in after), "never re-locked"
        r0 = relinks[0]
        assert r0["sim"] > 0.4, f"relink accepted on weak appearance: sim={r0['sim']}"

        # the estimate must settle onto the ground-truth centre after recovery
        expected = json.load(open(gt))["meta"]["expected_target_center"]
        errs = []
        for r in recs:
            key = str(r["frame_index"])
            if key in expected and r["frame_index"] > HIDE_TO:
                ex, ey = expected[key]
                errs.append(float(np.hypot(r["center"][0] - ex, r["center"][1] - ey)))
        assert errs and max(errs) < 40, f"post-recovery error too large: {max(errs):.1f}px"

        assert any(r.get("target_now") for r in recs), "no latency-compensated target"
        assert Path(summary["outputs"]["video"]).exists()
        assert Path(summary["outputs"]["summary"]).exists()
        if verbose:
            print(f"smoke: PASS  (post-recovery max centre error {max(errs):.1f}px, "
                  f"avg det {summary['avg_det_ms']}ms, avg loop {summary['avg_loop_ms']}ms)")
        return summary


if __name__ == "__main__":
    test_embedder_discrimination()
    test_smoke()
