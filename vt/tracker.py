"""Primary-subject lock.

One object, selected once by the operator, is followed for the rest of the clip:

  LOCKED    -- detection matched last frame: Kalman fused, templates updated
  COASTING  -- 1..coast_max frames with no match: constant-velocity prediction
               (+ optional LK flow refinement), NO template updates
  LOST      -- coast_max..lost_max frames: full-frame low-confidence re-detect,
               relaxed gates, re-link by EMA/gallery embedding + motion prior
  DEAD      -- past lost_max: lock abandoned

Template-drift control (spec 2.4): a fixed anchor embedding from the selection
frame is never overwritten; the EMA copy and a small gallery only absorb new
appearances while the update gate passes. If the target's appearance diverges
from the anchor past `drift_floor` (extreme lighting change, rotation, partial
occlusion), updates freeze instead of avalanching into a corrupted template.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from .geometry import (box_area, box_center, box_size, center_to_box, clip_box,
                       frame_diagonal, iou_xyxy, ipd_depth, pseudo_depth_from_height,
                       search_radius)
from .gmc import GlobalMotion
from .kalman import KalmanCV
from .reid import cos_sim, l2norm

STATE_LOCKED = "LOCKED"
STATE_COASTING = "COASTING"
STATE_LOST = "LOST"
STATE_DEAD = "DEAD"


@dataclass
class LockReading:
    state: str
    frame_index: int
    t: float
    bbox: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    depth_m: Optional[float] = None
    depth_src: str = ""
    track_id: Optional[int] = None
    score: float = 0.0
    sim: float = 0.0
    sim_anchor: float = 0.0
    iou: float = 0.0
    maha: float = 0.0
    misses: int = 0
    measured: bool = False
    relinked: bool = False
    template_updates: int = 0
    template_freezes: int = 0
    face_hits: int = 0
    traffic: float = 0.0

    def as_dict(self):
        d = asdict(self)
        for k in ("bbox", "center", "velocity"):
            d[k] = [float(v) for v in np.asarray(d[k]).ravel()]
        return d


class IdAssigner:
    """Greedy IoU association so track IDs persist even when no BoT-SORT is in use
    (ONNX / TFLite deployment path). With ultralytics the real BoT-SORT IDs are
    used and this is skipped."""

    def __init__(self, iou_min=0.3, max_age=30):
        self.iou_min = float(iou_min)
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks = {}

    def assign(self, dets):
        for tid in self.tracks:
            self.tracks[tid][1] += 1
        used = set()
        for d in dets:
            if getattr(d, "track_id", None) is not None:
                self.tracks[d.track_id] = [np.asarray(d.xyxy, dtype=np.float64).copy(), 0]
                used.add(d.track_id)
                self.next_id = max(self.next_id, int(d.track_id) + 1)
                continue
            best, best_iou = None, self.iou_min
            for tid, (box, _age) in self.tracks.items():
                if tid in used:
                    continue
                v = iou_xyxy(d.xyxy, box)
                if v > best_iou:
                    best, best_iou = tid, v
            tid = best if best is not None else self.next_id
            if best is None:
                self.next_id += 1
            else:
                used.add(tid)
            self.tracks[tid] = [np.asarray(d.xyxy, dtype=np.float64).copy(), 0]
            d.track_id = tid
        for tid in [t for t, (_b, a) in self.tracks.items() if a > self.max_age]:
            del self.tracks[tid]
        return dets


@dataclass
class Candidate:
    det: object
    emb: Optional[np.ndarray]
    iou: float
    sim: float
    sim_anchor: float
    maha: float
    dist: float
    id_match: bool
    cost: float


class SubjectTracker:
    def __init__(self, tracker_cfg, reid_cfg, depth_cfg, face_cfg, embedder,
                 frame, bbox, t, frame_index=0, track_id=None, score=1.0, cls_id=0,
                 face_probe=None, camera=None, flow_refiner=None):
        self.cfg = tracker_cfg
        self.reid_cfg = reid_cfg
        self.depth_cfg = depth_cfg
        self.face_cfg = face_cfg or {}
        self.embedder = embedder
        self.face_probe = face_probe
        self.camera = camera or {}
        self.flow = flow_refiner

        bbox = np.asarray(bbox, dtype=np.float64)
        center = box_center(bbox)
        size = box_size(bbox)
        self.cls_id = cls_id
        self.frame_index = int(frame_index)
        self.last_t = float(t)
        self.start_t = float(t)
        self.state = STATE_LOCKED
        self.misses = 0
        self.track_id = track_id
        self.score = float(score)
        self.size = np.array(size, dtype=np.float64)
        self.size_alpha = float(self.cfg.get("size_alpha", 0.25))
        self.trail = deque(maxlen=int(self.cfg.get("trail_len", 180)))
        self.history = []

        self.use_depth = bool(self.depth_cfg.get("enable", True)) and \
            str(self.depth_cfg.get("mode", "ipd")) != "off"
        dim = 3 if self.use_depth else 2
        self.kf = KalmanCV(dim=dim, process_noise=self.cfg.get("process_noise", 40.0),
                           meas_noise=self.cfg.get("meas_noise", 3.0),
                           pos=(center[0], center[1], 0.0), vel=(0.0, 0.0, 0.0))

        self.anchor = None
        self.ema = None
        self.gallery = deque(maxlen=int(reid_cfg.get("gallery_size", 8)))
        emb = self.embedder.extract(frame, bbox)
        if emb is not None:
            self.anchor = emb.copy()
            self.ema = emb.copy()
            self.gallery.append(emb.copy())

        self.face_hits = 0
        self.template_updates = 0
        self.template_freezes = 0
        self.relinks = 0
        self.last_depth = None
        self.last_face = None
        self.counters = {"LOCKED": 0, "COASTING": 0, "LOST": 0, "DEAD": 0}
        self.trail.append((float(t), center.copy(), self.size.copy()))
        if self.flow is not None:
            try:
                self.flow.seed(frame, bbox)
            except Exception:
                self.flow = None

    # ------------------------------------------------------------------ helpers
    def best_sim(self, emb):
        if emb is None or self.ema is None:
            return 0.0, 0.0
        sims = [cos_sim(emb, self.ema)]
        if self.anchor is not None:
            sims.append(cos_sim(emb, self.anchor))
        for g in self.gallery:
            sims.append(cos_sim(emb, g))
        return float(max(sims)), float(cos_sim(emb, self.anchor))

    def _predicted_size(self):
        w, h = float(self.size[0]), float(self.size[1])
        if self.state != STATE_LOCKED:
            g = float(self.cfg.get("coast_size_growth", 0.02))
            w *= 1.0 + g * min(self.misses, int(self.cfg.get("coast_max_frames", 30)))
            h *= 1.0 + g * min(self.misses, int(self.cfg.get("coast_max_frames", 30)))
        return np.array([max(w, 2.0), max(h, 2.0)], dtype=np.float64)

    def _measure_depth(self, frame, bbox):
        if not self.use_depth:
            return None, ""
        mode = str(self.depth_cfg.get("mode", "ipd"))
        focal = float(self.camera.get("focal_px") or 0.0) or None
        if mode == "pseudo":
            z = pseudo_depth_from_height(box_size(bbox)[1], self.depth_cfg.get("person_height_m", 1.7), focal)
            return z, "pseudo" if z else ""
        if self.face_probe is None or not self.face_probe.enable:
            return None, ""
        faces = self.face_probe.detect(frame, roi=bbox)
        face = self.face_probe.best_for_box(faces, bbox)
        if face is None:
            return None, ""
        self.last_face = face
        self.face_hits += 1
        z = ipd_depth(face["ipd_px"], focal, self.depth_cfg.get("ipd_m", 0.063))
        return z, "ipd" if z else ""

    # --------------------------------------------------------------------- step
    def step(self, frame, dets, t, frame_index=0, M=None, redetect: Optional[Callable] = None):
        self.frame_index = int(frame_index)
        dt = float(t) - self.last_t
        self.kf.predict(dt)
        if M is not None and self.state != STATE_LOCKED:
            GlobalMotion.apply_to_state(self.kf, M)

        pred_center = self.kf.position()[:2]
        pred_size = self._predicted_size()
        pred_bbox = clip_box(center_to_box(pred_center, pred_size),
                             frame.shape[1], frame.shape[0])

        # The subject may have moved outside the current detections (occlusion).
        # Only when LOST do we pay for a full-frame low-confidence re-detect.
        if self.state == STATE_LOST and redetect is not None:
            every = int(self.cfg.get("redetect_every", 5))
            if every > 0 and self.misses % every == 0:
                extra = redetect(frame, frame_index)
                if extra:
                    dets = list(dets) + list(extra)

        radius = search_radius(pred_bbox, base_px=self.cfg.get("search_base_px", 28.0),
                               growth=self.cfg.get("search_growth", 0.25),
                               misses=self.misses,
                               diag_frac=self.cfg.get("search_diag_frac", 0.35),
                               frame_diag=frame_diagonal(frame.shape))
        best = self._match(frame, dets, pred_bbox, pred_center, radius)

        depth_z, depth_src = self._measure_depth(frame, pred_bbox)
        if depth_z is not None:
            self.last_depth = depth_z

        relinked = False
        measured = False
        if best is not None:
            if self.misses > 0:
                relinked = True
                self.relinks += 1
            self.state = STATE_LOCKED
            self.misses = 0
            measured = True
            self.kf.update(best.det.center(), dims=(0, 1))
            if depth_z is not None and self.kf.dim >= 3:
                self.kf.update([depth_z], dims=(2,))
            new_size = np.array(box_size(best.det.xyxy), dtype=np.float64)
            a = self.size_alpha
            self.size = (1.0 - a) * self.size + a * new_size
            self.score = float(best.det.score)
            if best.det.track_id is not None:
                self.track_id = best.det.track_id
            self._maybe_update_templates(best)
            if self.flow is not None:
                try:
                    self.flow.seed(frame, best.det.xyxy)
                except Exception:
                    self.flow = None
        else:
            self.misses += 1
            self.kf.scale_uncertainty(float(self.cfg.get("uncertainty_growth", 1.12)))
            coast_max = int(self.cfg.get("coast_max_frames", 30))
            lost_max = int(self.cfg.get("lost_max_frames", 240))
            if self.misses <= coast_max:
                self.state = STATE_COASTING
            elif self.misses <= lost_max:
                self.state = STATE_LOST
            else:
                self.state = STATE_DEAD
            if self.flow is not None and self.state == STATE_COASTING:
                try:
                    refined, conf = self.flow.refine(frame, pred_bbox)
                    if refined is not None and conf > 0.35:
                        self.kf.reset_position(box_center(refined))
                        self.size = (1.0 - self.size_alpha) * self.size + \
                            self.size_alpha * np.array(box_size(refined), dtype=np.float64)
                except Exception:
                    self.flow = None

        read_center = self.kf.position()[:2]
        read_size = self._predicted_size()
        read_bbox = clip_box(center_to_box(read_center, read_size), frame.shape[1], frame.shape[0])
        self.last_t = float(t)
        self.trail.append((float(t), read_center.copy(), read_size.copy()))
        self.counters[self.state] = self.counters.get(self.state, 0) + 1

        reading = LockReading(
            state=self.state, frame_index=self.frame_index, t=float(t), bbox=read_bbox,
            center=read_center, velocity=self.kf.velocity()[:2],
            depth_m=(depth_z if depth_z is not None else self.last_depth),
            depth_src=depth_src or ("held" if self.last_depth else ""),
            track_id=self.track_id, score=self.score,
            sim=best.sim if best else 0.0, sim_anchor=best.sim_anchor if best else 0.0,
            iou=best.iou if best else 0.0, maha=best.maha if best else 0.0,
            misses=self.misses, measured=measured, relinked=relinked,
            template_updates=self.template_updates, template_freezes=self.template_freezes,
            face_hits=self.face_hits)
        self.history.append(reading)
        return reading

    # ------------------------------------------------------------------ matching
    def _match(self, frame, dets, pred_bbox, pred_center, radius):
        if not dets:
            return None
        self._last_radius = radius
        best: Optional[Candidate] = None
        embedder_off = str(getattr(self.embedder, "name", "")) == "none"
        # State-dependent gates: stricter while LOCKED (do not hand the lock to a
        # lookalike), looser while coasting or lost (accept a partially occluded
        # or blurred view of the same subject).
        locked = self.state == STATE_LOCKED
        iou_thr = float(self.cfg.get("iou_min", 0.20)) if locked \
            else float(self.cfg.get("iou_min_coast", 0.02))
        sim_thr = float(self.cfg.get("sim_min", 0.50)) if locked \
            else float(self.cfg.get("sim_min_lock", 0.40))
        maha_gate = float(self.cfg.get("maha_gate", 4.0))
        hard_floor = float(self.cfg.get("hard_sim_floor", 0.30))
        strong_iou = float(self.cfg.get("iou_strong", 0.50))
        for d in dets:
            if self.cls_id is not None and d.cls_id != self.cls_id:
                continue
            if box_area(d.xyxy) < float(self.cfg.get("min_area_px", 16.0)):
                continue
            emb = None if embedder_off else self.embedder.extract(frame, d.xyxy)
            iou = iou_xyxy(pred_bbox, d.xyxy)
            sim, sim_anchor = (1.0, 1.0) if embedder_off else self.best_sim(emb)
            maha = self.kf.mahalanobis(d.center(), dims=(0, 1))
            dist = float(np.linalg.norm(d.center() - pred_center))
            id_match = (self.track_id is not None and d.track_id == self.track_id)

            spatial_ok = (iou >= iou_thr) or (maha <= maha_gate) or (dist <= radius) or id_match
            app_ok = embedder_off or (sim >= sim_thr)
            plausible = (not embedder_off) and sim < hard_floor and not id_match
            if embedder_off:
                accept = spatial_ok
            else:
                accept = (app_ok and spatial_ok) or (iou >= strong_iou and sim >= hard_floor)
                if plausible:
                    accept = False

            cost = (float(self.cfg.get("w_iou", 1.0)) * (1.0 - iou)
                    + float(self.cfg.get("w_emb", 2.0)) * (1.0 - sim)
                    + float(self.cfg.get("w_motion", 0.5)) * min(maha / max(maha_gate, 1e-6), 1.5)
                    - float(self.cfg.get("w_id", 0.8)) * (1.0 if id_match else 0.0))
            cand = Candidate(det=d, emb=emb, iou=iou, sim=sim, sim_anchor=sim_anchor,
                             maha=maha, dist=dist, id_match=id_match, cost=cost)
            if not accept:
                continue
            if best is None or cand.cost < best.cost:
                best = cand
        return best

    def _maybe_update_templates(self, cand: Candidate):
        """Score-based gating + drift guard (spec 2.4)."""
        if self.ema is None or cand.emb is None:
            return
        score_gate = float(self.reid_cfg.get("update_score_gate", 0.50))
        sim_gate = float(self.reid_cfg.get("update_sim_min", 0.55))
        drift_floor = float(self.reid_cfg.get("drift_floor", 0.40))
        freeze = False
        if cand.det.score < score_gate:
            freeze = True
        elif cand.sim < sim_gate:
            freeze = True
        elif self.anchor is not None and cand.sim_anchor < drift_floor:
            freeze = True  # target appearance diverged: do NOT absorb this view
        if freeze:
            self.template_freezes += 1
            return
        alpha = float(self.reid_cfg.get("ema_alpha", 0.10))
        self.ema = l2norm((1.0 - alpha) * self.ema + alpha * cand.emb)
        dup = max((cos_sim(cand.emb, g) for g in self.gallery), default=-1.0)
        if dup < float(self.reid_cfg.get("gallery_dedupe", 0.98)):
            self.gallery.append(cand.emb.copy())
        self.template_updates += 1

    # ------------------------------------------------------------------ reporting
    def summary(self):
        return {
            "states": {k: int(v) for k, v in self.counters.items()},
            "template_updates": self.template_updates,
            "template_freezes": self.template_freezes,
            "relinks": self.relinks,
            "face_hits": self.face_hits,
            "final_state": self.state,
            "track_id": self.track_id,
            "gallery_size": len(self.gallery),
        }
