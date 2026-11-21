"""Lucas-Kanade optical-flow refinement of the locked box.

Runs only on the primary subject's region of interest, so it costs almost
nothing. It gives sub-detection-frame box updates (useful at 30 Hz when the
motor runs at 1 kHz) and keeps the box glued to the target for the first few
coasting frames instead of trusting pure constant-velocity extrapolation.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .geometry import box_center, box_size, center_to_box, ellipse_mask


class LKRefiner:
    def __init__(self, enable=True, max_points=60, quality=0.01, min_distance=4,
                 win=(15, 15), levels=3, max_shift_frac=0.5, min_points=4):
        self.enable = bool(enable) and cv2 is not None
        self.max_points = int(max_points)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.win = tuple(win)
        self.levels = int(levels)
        self.max_shift_frac = float(max_shift_frac)
        self.min_points = int(min_points)
        self.prev_gray = None
        self._pts = None
        self._roi = None

    def _prep(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return g

    def seed(self, frame, bbox, mask="ellipse"):
        if not self.enable:
            return
        gray = self._prep(frame)
        m = ellipse_mask(gray.shape, box_center(bbox), box_size(bbox), pad=0.05)
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, gray.shape[1]), min(y2, gray.shape[0])
        if x2 - x1 < 3 or y2 - y1 < 3:
            self._pts = None
            return
        sub = gray[y1:y2, x1:x2]
        subm = m[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(sub, maxCorners=self.max_points,
                                      qualityLevel=self.quality,
                                      minDistance=self.min_distance, mask=subm)
        self.prev_gray = gray
        if pts is None or len(pts) < self.min_points:
            self._pts = None
            self._roi = None
            return
        pts = pts.reshape(-1, 2) + np.array([x1, y1], dtype=np.float32)
        self._pts = pts.astype(np.float32).reshape(-1, 1, 2)
        self._roi = (x1, y1, x2, y2)

    def refine(self, frame, bbox):
        """Returns (refined_bbox, confidence 0..1) or (None, 0.0) when flow failed."""
        if not self.enable or self._pts is None or self.prev_gray is None:
            return None, 0.0
        gray = self._prep(frame)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self._pts, None,
                                             winSize=self.win, maxLevel=self.levels)
        self.prev_gray = gray
        if p1 is None or st is None:
            self._pts = None
            return None, 0.0
        ok = st.reshape(-1).astype(bool)
        a = self._pts.reshape(-1, 2)[ok]
        b = p1.reshape(-1, 2)[ok]
        if len(a) < self.min_points:
            self._pts = None
            return None, 0.0
        disp = b - a
        med = np.median(disp, axis=0)
        w, h = box_size(bbox)
        if float(np.linalg.norm(med)) > self.max_shift_frac * max(w, h):
            self._pts = None
            return None, 0.0  # implausible jump, let Kalman own this frame
        c = box_center(bbox) + med
        new_bbox = center_to_box(c, (w, h))
        # Re-seed every frame to stop point drift accumulation.
        self.seed(frame, new_bbox)
        conf = float(min(1.0, len(a) / max(self.max_points, 1)))
        return new_bbox, conf

    def bump(self, frame, bbox, offset):
        self.seed(frame, bbox)
