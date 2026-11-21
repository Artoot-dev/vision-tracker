"""Global Motion Compensation (GMC).

BoT-SORT's GMC step exists to remove camera / gimbal motion before associating
detections to tracks. When the ultralytics BoT-SORT tracker is used, its own GMC
runs inside the tracker; this implementation is what covers the ONNX/TFLite
deployment path and any custom association we do, plus it feeds the Kalman state
so a panning camera does not look like a target that teleported.

Sparse LK flow + RANSAC affine (partial: rotation + uniform scale + translation).
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class GlobalMotion:
    def __init__(self, enable=True, max_corners=150, quality=0.01, min_distance=12,
                 downscale=2, min_inliers=8, max_shift_px=None):
        self.enable = bool(enable) and cv2 is not None
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.downscale = max(int(downscale), 1)
        self.min_inliers = int(min_inliers)
        self.max_shift_px = max_shift_px
        self.M = None
        self.prev_gray = None
        self.last_info = {}

    def _gray(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            g = cv2.resize(g, (g.shape[1] // self.downscale, g.shape[0] // self.downscale),
                           interpolation=cv2.INTER_AREA)
        return g

    def update(self, frame):
        """Call once per frame, in order. Returns the 3x2 affine M (prev -> cur)
        in full-resolution coordinates, or None when GMC is unavailable."""
        if not self.enable:
            return None
        gray = self._gray(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.M = None
            return None

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=self.max_corners,
                                     qualityLevel=self.quality,
                                     minDistance=self.min_distance)
        M = None
        inliers = 0
        if p0 is not None and len(p0) >= 4:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
            if p1 is not None:
                ok = st.reshape(-1).astype(bool)
                a = p0.reshape(-1, 2)[ok]
                b = p1.reshape(-1, 2)[ok]
                if len(a) >= 4:
                    M, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                          ransacReprojThreshold=3.0)
                    if mask is not None:
                        inliers = int(mask.sum())
        self.prev_gray = gray

        if M is None or inliers < self.min_inliers:
            self.M = None
            self.last_info = {"ok": False, "inliers": inliers}
            return None

        s = float(self.downscale)
        M = M.copy()
        M[0, 2] *= s
        M[1, 2] *= s
        # Reject absurd transforms (scene cut, full occlusion) instead of warping into the void.
        shift = float(np.hypot(M[0, 2], M[1, 2]))
        scale = float(np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])))
        limit = self.max_shift_px
        if limit is not None and shift > limit:
            self.M = None
            self.last_info = {"ok": False, "reason": "shift", "shift": shift, "inliers": inliers}
            return None
        if not (0.85 <= scale <= 1.18):
            self.M = None
            self.last_info = {"ok": False, "reason": "scale", "scale": scale, "inliers": inliers}
            return None

        self.M = M
        self.last_info = {"ok": True, "inliers": inliers, "shift_px": shift, "scale": scale}
        return M

    @staticmethod
    def apply_point(M, pt):
        if M is None:
            return np.asarray(pt, dtype=np.float64)
        p = np.array([float(pt[0]), float(pt[1]), 1.0])
        return np.array([M[0] @ p, M[1] @ p], dtype=np.float64)

    @staticmethod
    def apply_box(M, xyxy):
        if M is None:
            return np.asarray(xyxy, dtype=np.float64)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x1, y2, 1.0], [x2, y2, 1.0]]).T
        out = (M @ pts).T
        return np.array([out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max()],
                        dtype=np.float64)

    @staticmethod
    def apply_to_state(kf, M):
        """Rotate/translate the Kalman position + velocity so a moving camera does
        not get mistaken for target motion. Only position delta and rotation of
        velocity are applied (scale is left on the box, not the state)."""
        if M is None or kf is None:
            return
        pos = kf.position()
        vel = kf.velocity()
        new_pos = GlobalMotion.apply_point(M, pos[:2])
        p2 = np.array([pos[0] + vel[0], pos[1] + vel[1]])
        new_p2 = GlobalMotion.apply_point(M, p2)
        new_vel = new_p2 - new_pos
        pos[0], pos[1] = new_pos[0], new_pos[1]
        if kf.dim >= 3:
            kf.x[3], kf.x[4] = new_vel[0], new_vel[1]
        else:
            kf.x[2], kf.x[3] = new_vel[0], new_vel[1]
        kf.x[:kf.dim] = pos
