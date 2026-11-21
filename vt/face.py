"""Face probe (dual-class detection: Person + Face).

Used for two things in the spec:
  - IPD-based monocular depth (face mesh / face box -> inter-pupillary pixels)
  - a second, independent cue that survives when the body box degrades

Uses the OpenCV bundled Haar cascade so there is no extra model file and no
MediaPipe/torch dependency on the device. Enable with face.enable: true.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# Average adult inter-pupillary distance; children ~0.05, adults 0.058-0.066.
REAL_IPD_M = 0.063
# IPD as a fraction of face-box width for the frontal Haar cascade.
IPD_FRAC_OF_FACE_W = 0.42


class FaceProbe:
    def __init__(self, enable=True, scale_factor=1.1, min_neighbors=5, min_size=(28, 28),
                 every_n=1, coarse=1.0):
        self.enable = bool(enable) and cv2 is not None
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(int(v) for v in min_size)
        self.every_n = max(int(every_n), 1)
        self._count = 0
        self._cache = []
        self.cascade = None
        if self.enable:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(path)
            if self.cascade.empty():
                self.enable = False
                self.cascade = None

    def detect(self, frame, roi=None):
        """Returns list of dicts: {xyxy, ipd_px, score}. roi = (x1,y1,x2,y2) limits
        the search to the subject box (far cheaper, and avoids faces of bystanders)."""
        if not self.enable:
            return []
        if self._count % self.every_n != 0:
            self._count += 1
            return self._cache
        self._count += 1
        h, w = frame.shape[:2]
        if roi is not None:
            x1, y1, x2, y2 = [int(max(0, v)) for v in roi]
            x2, y2 = min(x2, w), min(y2, h)
            if x2 - x1 < 16 or y2 - y1 < 16:
                self._cache = []
                return []
            sub = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
        else:
            sub, ox, oy = frame, 0, 0
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=self.scale_factor,
                                              minNeighbors=self.min_neighbors,
                                              minSize=self.min_size)
        out = []
        for (fx, fy, fw, fh) in faces:
            out.append({
                "xyxy": np.array([fx + ox, fy + oy, fx + fw + ox, fy + fh + oy], dtype=np.float64),
                "ipd_px": float(fw * IPD_FRAC_OF_FACE_W),
                "score": 1.0,
            })
        self._cache = out
        return out

    @staticmethod
    def best_for_box(faces, bbox):
        """Pick the face whose box sits inside / nearest to the subject box (the
        subject's own face, not a bystander)."""
        if not faces:
            return None
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        best, best_d = None, None
        for f in faces:
            fx1, fy1, fx2, fy2 = f["xyxy"]
            fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            inside = (bx1 <= fcx <= bx2) and (by1 <= fcy <= by2)
            d = np.hypot(fcx - bcx, fcy - bcy) * (0.5 if inside else 1.0)
            if best_d is None or d < best_d:
                best, best_d = f, d
        return best
