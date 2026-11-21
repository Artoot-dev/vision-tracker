"""Appearance feature extraction for online re-ID.

Two backends:

lite (default)  Pure NumPy/OpenCV descriptor, ~85 dims, <1 ms on a 320px crop,
                no model file at all. Built for edge devices where a second
                CNN on top of the detector is not affordable.
mobilenet       ONNX MobileNetV3-Small penultimate features (576 dims).
                Swap in when you have the compute budget.

Both return L2-normalised float32 vectors, so cosine similarity is a dot product.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def cos_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def l2norm(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


class LiteEmbedder:
    """Center-weighted colour + spatial + shape descriptor.

    Four blocks, each unit-normalised then scaled by `block_weights`, then the
    whole vector is L2-normalised:
      [0] hue/saturation histogram (saturation-gated, so neutral greys do not
          inject meaningless hue)
      [1] 2x2 spatial means per channel, mean-subtracted
      [2] 3x3 pooled gradient magnitude, mean-subtracted (mean-centering is what
          gives this block any discriminative power; without it every target
          collapses to ~the same all-ones direction and everything scores 0.95)
      [3] value histogram (keeps a colourless target, e.g. a white drone against
          sky, from producing an all-zero descriptor)
    """

    name = "lite"

    def __init__(self, mask="ellipse", crop_pad=0.0, size=(32, 64), hist_h=8, hist_s=8,
                 sat_min=25, block_weights=(2.0, 1.0, 1.0, 1.0)):
        self.mask = mask
        self.crop_pad = float(crop_pad)
        self.size = (int(size[0]), int(size[1]))
        self.hist_h = int(hist_h)
        self.hist_s = int(hist_s)
        self.sat_min = float(sat_min)
        self.block_weights = tuple(float(w) for w in (block_weights or (2.0, 1.0, 1.0, 1.0)))
        self.dim = self.hist_h * self.hist_s + 12 + 9 + 8

    def _weights(self, w, h):
        if self.mask == "none":
            return np.ones((h, w), dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - (w - 1) / 2.0) / max(w / 2.0, 1e-6)
        ny = (yy - (h - 1) / 2.0) / max(h / 2.0, 1e-6)
        r2 = nx * nx + ny * ny
        if self.mask == "ellipse":
            return np.clip(1.0 - r2, 0.0, 1.0) ** 0.5
        return np.exp(-2.0 * r2)  # soft center weighting

    def crop(self, frame, xyxy):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw, bh = x2 - x1, y2 - y1
        pad = self.crop_pad * max(bw, bh)
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]

    def extract(self, frame, xyxy):
        if cv2 is None:
            return None
        crop = self.crop(frame, xyxy)
        if crop is None:
            return None
        cw, ch = self.size
        small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA)
        wt = self._weights(cw, ch)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # [0] saturation-gated weighted H/S histogram. np.bincount does this
        # directly: cv2.calcHist only takes a binary mask, which would throw the
        # spatial weighting away.
        satw = (hsv[:, :, 1] > self.sat_min).astype(np.float32)
        wts = wt * satw
        hb = np.clip(hsv[:, :, 0].astype(np.int32) * self.hist_h // 180, 0, self.hist_h - 1)
        sb = np.clip(hsv[:, :, 1].astype(np.int32) * self.hist_s // 256, 0, self.hist_s - 1)
        hist = np.bincount((hb * self.hist_s + sb).ravel(), weights=wts.ravel(),
                           minlength=self.hist_h * self.hist_s).astype(np.float32)
        total = float(hist.sum())
        if total > 1e-6:
            hist /= total

        # [1] spatial colour layout, mean-subtracted so it measures arrangement,
        # not just overall brightness
        bgr = small.astype(np.float32)
        cells = []
        for cy in (0, ch // 2):
            for cx in (0, cw // 2):
                block = bgr[cy:cy + ch // 2, cx:cx + cw // 2]
                cells.append(block.reshape(-1, 3).mean(axis=0))
        cells = np.asarray(cells, dtype=np.float32)
        cells = (cells - cells.mean(axis=0, keepdims=True)) / 128.0

        # [2] mean-centered pooled gradient magnitude
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy).mean(axis=2)
        pooled = cv2.resize(mag, (3, 3), interpolation=cv2.INTER_AREA).ravel()
        pooled = pooled - pooled.mean()

        # [3] value histogram (weighted by the spatial mask only, no saturation gate)
        vb = np.clip(hsv[:, :, 2].astype(np.int32) * 8 // 256, 0, 7)
        vhist = np.bincount(vb.ravel(), weights=wt.ravel(), minlength=8).astype(np.float32)
        vt = float(vhist.sum())
        if vt > 1e-6:
            vhist /= vt

        blocks = (hist, cells.ravel(), pooled, vhist)
        parts = [l2norm(b) * w for b, w in zip(blocks, self.block_weights)]
        return l2norm(np.concatenate(parts))


class MobileNetEmbedder:
    """ONNX MobileNetV3-Small feature extractor (576-d)."""

    name = "mobilenet"

    def __init__(self, weights, size=(224, 224), mask="ellipse", crop_pad=0.05):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = (int(size[0]), int(size[1]))
        self.mask = mask
        self.crop_pad = float(crop_pad)
        self.dim = 576
        self._shape = shape

    def extract(self, frame, xyxy):
        if cv2 is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        pad = self.crop_pad * max(x2 - x1, y2 - y1)
        x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        x2, y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        crop = cv2.resize(frame[y1:y2, x1:x2], self.size, interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(crop, scalefactor=1 / 255.0 / 0.226,
                                     size=self.size,
                                     mean=(0.485 / 0.226, 0.456 / 0.226, 0.406 / 0.226),
                                     swapRB=True, crop=False)
        out = self.sess.run(None, {self.inp: blob})[0]
        v = np.asarray(out, dtype=np.float32).ravel()
        return l2norm(v)


class NoEmbedder:
    name = "none"
    dim = 0

    def extract(self, frame, xyxy):
        return None


def make_embedder(cfg):
    kind = str(cfg.get("embedder", "lite")).lower()
    if kind in ("none", "off", "false"):
        return NoEmbedder()
    if kind == "mobilenet":
        w = cfg.get("weights")
        if not w:
            raise ValueError("reid.weights (ONNX path) is required for embedder=mobilenet")
        return MobileNetEmbedder(w, size=cfg.get("mobilenet_size", (224, 224)),
                                 mask=cfg.get("mask", "ellipse"),
                                 crop_pad=cfg.get("crop_pad", 0.05))
    return LiteEmbedder(mask=cfg.get("mask", "ellipse"), crop_pad=cfg.get("crop_pad", 0.0),
                        size=cfg.get("size", (32, 64)), hist_h=cfg.get("hist_h", 8),
                        hist_s=cfg.get("hist_s", 8), sat_min=cfg.get("sat_min", 25),
                        block_weights=cfg.get("block_weights", (2.0, 1.0, 1.0, 1.0)))
