"""Box math, pinhole/IPD depth, and spatial reliability helpers."""
from __future__ import annotations

import math

import numpy as np


def box_center(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def box_size(xyxy):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return float(max(x2 - x1, 1.0)), float(max(y2 - y1, 1.0))


def box_area(xyxy):
    w, h = box_size(xyxy)
    return w * h


def center_to_box(center, size):
    cx, cy = float(center[0]), float(center[1])
    w, h = float(size[0]), float(size[1])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float64)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_box(pt, xyxy):
    x, y = float(pt[0]), float(pt[1])
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1 <= x <= x2 and y1 <= y <= y2


def clip_box(xyxy, w, h):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return np.array([min(max(x1, 0.0), w - 1.0), min(max(y1, 0.0), h - 1.0),
                     min(max(x2, 0.0), w - 1.0), min(max(y2, 0.0), h - 1.0)], dtype=np.float64)


# --------------------------------------------------------------------- depth

def focal_px_from_hfov(width_px, hfov_deg):
    return float(width_px) / (2.0 * math.tan(math.radians(float(hfov_deg)) / 2.0))


def focal_px_from_sensor(width_px, sensor_w_mm, focal_mm):
    return float(width_px) * (float(focal_mm) / float(sensor_w_mm))


def ipd_depth(ipd_px, focal_px, real_ipd_m=0.063):
    """Monocular pinhole depth from inter-pupillary distance.

    z = f * X_real / x_pixels
    """
    ipd_px = float(ipd_px)
    if ipd_px < 1e-3 or focal_px is None or focal_px <= 0:
        return None
    return float(float(focal_px) * float(real_ipd_m) / ipd_px)


def pseudo_depth_from_height(bbox_h_px, real_h_m, focal_px):
    """Depth from a known real-world height (e.g. 1.7 m person). Scale-ambiguous
    across different subjects, so treat as a coarse prior only."""
    if bbox_h_px <= 1e-3 or not focal_px:
        return None
    return float(float(focal_px) * float(real_h_m) / float(bbox_h_px))


def idt_scale(prev_bbox, cur_bbox):
    """Isotropic scale change between two boxes (used for scale adaptation)."""
    pw, ph = box_size(prev_bbox)
    cw, ch = box_size(cur_bbox)
    return float(math.sqrt(max((cw / pw) * (ch / ph), 1e-6)))


def search_radius(bbox, base_px=28.0, growth=0.25, misses=0, diag_frac=0.35, frame_diag=None):
    """Expanding search window radius for re-acquisition while coasting."""
    w, h = box_size(bbox)
    r = float(base_px) + float(growth) * float(misses) * max(w, h) * 0.5
    if frame_diag:
        r = min(r, float(diag_frac) * float(frame_diag))
    return r


def ellipse_mask(shape, center, size, pad=0.15):
    """Non-rectangular reliability mask (spec 2.2 spatial reliability mapping)."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = float(center[0]), float(center[1])
    rx = max(float(size[0]) / 2.0 * (1.0 + pad), 1.0)
    ry = max(float(size[1]) / 2.0 * (1.0 + pad), 1.0)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return (d <= 1.0).astype(np.uint8)


def frame_diagonal(shape):
    h, w = shape[:2]
    return float(math.hypot(w, h))
