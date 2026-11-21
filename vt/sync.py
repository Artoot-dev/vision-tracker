"""Time sync / latency decoupling.

The visual front end runs at 30 Hz; the motor loop runs at 1 kHz. This module
owns the bridge:

  TrajectoryBuffer  timestamped position/velocity samples -> interpolate (inside
                    the sampled window) or constant-velocity extrapolate (a short
                    way past the last sample)
  LatencyModel      running estimate of end-to-end pipeline latency, so the pose
                    handed to the motor is predicted to the moment the motor
                    actually acts, not the moment the frame was captured
"""
from __future__ import annotations

from collections import deque

import numpy as np


class TrajectoryBuffer:
    def __init__(self, maxlen=900):
        self.samples = deque(maxlen=int(maxlen))

    def push(self, t, pos, vel=None, state="", bbox=None):
        pos = np.asarray(pos, dtype=np.float64).ravel()
        vel = np.zeros(3) if vel is None else np.asarray(vel, dtype=np.float64).ravel()
        if pos.size < 3:
            pos = np.concatenate([pos, np.zeros(3 - pos.size)])
        if vel.size < 3:
            vel = np.concatenate([vel, np.zeros(3 - vel.size)])
        self.samples.append({"t": float(t), "p": pos[:3].copy(), "v": vel[:3].copy(),
                             "state": state, "bbox": bbox})

    def __len__(self):
        return len(self.samples)

    @property
    def last(self):
        return self.samples[-1] if self.samples else None

    def sample(self, t, max_extrapolate_s=0.12):
        """Position at time t. Interpolates between bracketing samples; beyond the
        last sample it extrapolates on the last velocity, but never further than
        max_extrapolate_s (better to hold than to fly off)."""
        if not self.samples:
            return None
        t = float(t)
        s = list(self.samples)
        if t <= s[0]["t"]:
            return {"t": s[0]["t"], "p": s[0]["p"], "v": s[0]["v"],
                    "state": s[0]["state"], "extrapolated": False, "clamped": "before"}
        for i in range(1, len(s)):
            a, b = s[i - 1], s[i]
            if a["t"] <= t <= b["t"]:
                span = max(b["t"] - a["t"], 1e-9)
                u = (t - a["t"]) / span
                return {"t": t, "p": a["p"] * (1 - u) + b["p"] * u,
                        "v": a["v"] * (1 - u) + b["v"] * u,
                        "state": b["state"], "extrapolated": False, "clamped": None}
        last = s[-1]
        ahead = t - last["t"]
        if ahead > max_extrapolate_s:
            ahead = max_extrapolate_s
            clamped = "after"
        else:
            clamped = None
        p = last["p"] + last["v"] * ahead
        return {"t": last["t"] + ahead, "p": p, "v": last["v"], "state": last["state"],
                "extrapolated": True, "clamped": clamped}


class LatencyModel:
    def __init__(self, alpha=0.2, fixed_ms=0.0, warmup=10):
        self.alpha = float(alpha)
        self.fixed_ms = float(fixed_ms)
        self.warmup = int(warmup)
        self.n = 0
        self.ema_ms = 0.0

    def update(self, ms):
        self.n += 1
        if self.n == 1:
            self.ema_ms = float(ms)
        else:
            self.ema_ms = (1 - self.alpha) * self.ema_ms + self.alpha * float(ms)
        return self.ema_ms

    @property
    def ms(self):
        if self.n < self.warmup:
            return self.ema_ms
        return self.ema_ms + self.fixed_ms

    @property
    def s(self):
        return self.ms / 1000.0


def motor_stream(buffer, t_start, t_end, hz=1000.0, max_extrapolate_s=0.12):
    """Yield the pose stream a 1 kHz controller would consume. Used by
    tools/motor_stream.py to prove the interpolation path end to end."""
    dt = 1.0 / float(hz)
    t = float(t_start)
    while t <= t_end:
        s = buffer.sample(t, max_extrapolate_s=max_extrapolate_s)
        if s is not None:
            yield t, s
        t += dt
