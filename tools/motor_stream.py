#!/usr/bin/env python3
"""Replay a tracking record at motor rate.

Proves the latency-decoupled path: the vision front end runs at 30 Hz, the
controller polls at 1 kHz, and every poll is answered by interpolation between
vision samples (or by a bounded constant-velocity extrapolation when the poll
lands past the newest frame).

  python tools/motor_stream.py --jsonl out/myclip_track.jsonl --hz 1000 \
      --out out/myclip_motor.jsonl --max-seconds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vt.sync import LatencyModel, TrajectoryBuffer  # noqa: E402


def load(path):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--hz", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = whole clip")
    ap.add_argument("--extrapolate-ms", type=float, default=120.0)
    ap.add_argument("--latency-comp", action="store_true",
                    help="emit the pose predicted for the moment the motor acts")
    a = ap.parse_args()

    recs = load(a.jsonl)
    if not recs:
        raise SystemExit("empty record file")
    buf = TrajectoryBuffer()
    lat = LatencyModel()
    for r in recs:
        buf.push(r["t"], [r["center"][0], r["center"][1], r.get("depth_m") or 0.0],
                 [r["velocity"][0], r["velocity"][1], 0.0], state=r["state"])
        lat.update(r.get("latency_ms", 0.0))

    t0, t1 = recs[0]["t"], recs[-1]["t"]
    if a.max_seconds:
        t1 = min(t1, t0 + a.max_seconds)
    dt = 1.0 / a.hz
    n = 0
    n_ext = 0
    n_gap = 0
    out = open(a.out, "w") if a.out else None
    t = t0
    try:
        while t <= t1:
            s = buf.sample(t, max_extrapolate_s=a.extrapolate_ms / 1000.0)
            if s is None:
                break
            p = s["p"]
            rec = {"t": round(t, 6), "x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                   "z": round(float(p[2]), 4), "vx": round(float(s["v"][0]), 3),
                   "vy": round(float(s["v"][1]), 3), "state": s["state"],
                   "src": "extrapolated" if s["extrapolated"] else "interpolated"}
            n += 1
            if s["extrapolated"]:
                n_ext += 1
            if s.get("clamped"):
                n_gap += 1
            if out:
                out.write(json.dumps(rec) + "\n")
            t += dt
    finally:
        if out:
            out.close()
    vision_hz = 1.0 / max(np.median(np.diff([r["t"] for r in recs])), 1e-9)
    print(f"vision samples {len(recs)} at ~{vision_hz:.1f} Hz -> motor stream {n} samples "
          f"at {a.hz:.0f} Hz")
    print(f"  interpolated {n - n_ext}  extrapolated {n_ext}  held-at-limit {n_gap}")
    print(f"  mean recorded vision latency {lat.ema_ms:.1f} ms")
    if out:
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
