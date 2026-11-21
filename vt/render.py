"""Overlays: detections, the locked subject, trail, HUD, selection frame."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .tracker import STATE_LOCKED, STATE_COASTING, STATE_LOST, STATE_DEAD

STATE_COLOR = {
    STATE_LOCKED: (0, 230, 0),
    STATE_COASTING: (0, 170, 255),
    STATE_LOST: (0, 0, 255),
    STATE_DEAD: (80, 80, 200),
}


class Renderer:
    def __init__(self, cfg, names=None):
        self.cfg = cfg
        self.names = names or {}
        self.trail_len = int(cfg.get("draw_trail", 180))
        self.hud = bool(cfg.get("hud", True))
        self.points = bool(cfg.get("points", False))
        self.keypoints = bool(cfg.get("keypoints", False))
        self.all_tracks = bool(cfg.get("all_tracks", True))

    def _corner_box(self, img, bbox, color, thick=2, frac=0.22):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        dx, dy = int(w * frac), int(h * frac)
        for (px, py, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1),
                                 (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + sx * dx, py), color, thick, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + sy * dy), color, thick, cv2.LINE_AA)

    def draw(self, frame, dets, reading, trail, info=None):
        info = info or {}
        for d in dets:
            if d is None:
                continue
            if self.all_tracks:
                x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)
                if d.track_id is not None:
                    cv2.putText(frame, f"id{d.track_id}", (x1, max(y1 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
                                cv2.LINE_AA)
            if self.keypoints and d.extra.get("keypoints") is not None:
                kps = np.asarray(d.extra["keypoints"], dtype=np.float32)
                for kp in kps.reshape(-1, 2):
                    if kp[0] > 0 and kp[1] > 0:
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), -1,
                                   cv2.LINE_AA)

        if trail is not None and len(trail) > 1:
            pts = [(int(round(p[0])), int(round(p[1]))) for _t, p, _s in list(trail)[-self.trail_len:]]
            if self.points:
                for i, pt in enumerate(pts):
                    u = i / max(len(pts) - 1, 1)
                    col = (int(60 + 120 * u), int(220 * u), int(255 * (1 - u)))
                    cv2.circle(frame, pt, 2 if i % 4 else 3, col, -1, cv2.LINE_AA)
            else:
                # chunked polylines: same fading look, 4 draw calls instead of 180,
                # which matters when the frame budget is 33 ms
                n = len(pts)
                chunks = 4
                for c in range(chunks):
                    i0 = c * (n - 1) // chunks
                    i1 = (c + 1) * (n - 1) // chunks
                    seg = np.asarray(pts[i0:i1 + 1], dtype=np.int32)
                    if len(seg) < 2:
                        continue
                    u = i1 / max(n - 1, 1)
                    col = (int(60 + 120 * u), int(220 * u), int(255 * (1 - u)))
                    cv2.polylines(frame, [seg], False, col, 2, cv2.LINE_AA)

        if reading is not None:
            color = STATE_COLOR.get(reading.state, (255, 255, 255))
            self._corner_box(frame, reading.bbox, color, 3)
            cx, cy = int(round(reading.center[0])), int(round(reading.center[1]))
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
            vx, vy = reading.velocity[0], reading.velocity[1]
            if abs(vx) + abs(vy) > 1.0:
                cv2.arrowedLine(frame, (cx, cy), (int(cx + vx * 0.25), int(cy + vy * 0.25)),
                                color, 2, cv2.LINE_AA, tipLength=0.3)
            label = f"{reading.state} id{reading.track_id} {reading.sim:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(int(reading.bbox[1]) - 8, th + 4)
            cv2.rectangle(frame, (int(reading.bbox[0]), ty - th - 6),
                          (int(reading.bbox[0]) + tw + 8, ty + 2), color, -1)
            cv2.putText(frame, label, (int(reading.bbox[0]) + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if self.hud:
            lines = [
                f"frame {info.get('frame_index', 0)}  t {info.get('t', 0.0):.3f}s",
                f"state {reading.state if reading else '-'}  misses {reading.misses if reading else 0}",
                f"det {info.get('n_det', 0)}  det_ms {info.get('det_ms', 0.0):.1f}  "
                f"loop_ms {info.get('loop_ms', 0.0):.1f}",
                f"out_fps {info.get('fps', 0.0):.1f}  latency {info.get('latency_ms', 0.0):.0f}ms",
                f"sim {reading.sim if reading else 0.0:.3f}  iou {reading.iou if reading else 0.0:.2f}  "
                f"maha {reading.maha if reading else 0.0:.1f}",
                f"depth {('%.2fm' % reading.depth_m) if (reading and reading.depth_m) else '-'} "
                f"{reading.depth_src if reading else ''}",
            ]
            if info.get("gmc"):
                lines.append(f"gmc {info['gmc']}")
            if info.get("pred_t") is not None:
                lines.append(f"motor@+{info.get('latency_ms', 0):.0f}ms "
                             f"({info['pred_p'][0]:.0f},{info['pred_p'][1]:.0f})")
            y = 20
            for ln in lines:
                cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                            cv2.LINE_AA)
                y += 19
            extra = info.get("extra") or []
            if extra:
                y += 6
                for ln in extra:
                    cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3,
                                cv2.LINE_AA)
                    cv2.putText(frame, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 255),
                                1, cv2.LINE_AA)
                    y += 19
        return frame
