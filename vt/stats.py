"""Fallback accounting: how often the tracker needed the detector again.

The question this answers: BoT-SORT carries a track forward from its motion model,
but it can only create or re-create a track from a YOLO detection. Every frame
where the subject's box is missing from the tracker output is a frame the tracker
could not sustain on its own; every new track ID is a frame where it had to hand
the subject back to YOLO and start a fresh track.

Per frame, for the subject track:
  sustained  the subject came back from the tracker with an associated detection
             -> BoT-SORT carried it forward from YOLO normally
  coast      no box for the subject: the motion model (Kalman) supplied the pose.
             Within tracker.coast_max_frames the same track can still resume.
  lost       coast budget exhausted, so the pipeline asks YOLO again for a
             full-frame low-confidence re-detect (the literal fallback)
  reacquire  the subject reappeared after a gap of >= 1 coast/lost frame
  id_switch  it came back under a DIFFERENT track ID, meaning BoT-SORT did not
             sustain the track and had to create a new one

Counters that matter for detector load:
  redetect_calls  low-confidence full-frame YOLO passes the pipeline had to run
  track_births    tracks created after the first frame (each one is a re-init)
  detect_every    if the detector runs only every Nth frame, sustained/fallback
                  tells you how long the tracker holds on its own between calls
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gap:
    start_frame: int
    end_frame: int
    length: int
    kind: str = "coast"
    resumed_track_id: int | None = None
    id_switch: bool = False

    def as_dict(self):
        return {"start_frame": self.start_frame, "end_frame": self.end_frame,
                "length": self.length, "kind": self.kind,
                "resumed_track_id": self.resumed_track_id,
                "id_switch": self.id_switch}


@dataclass
class FallbackStats:
    window: int = 120

    def __post_init__(self):
        self.frames = 0
        self.sustained = 0
        self.coast = 0
        self.lost = 0
        self.dead = 0
        self.redetect_calls = 0
        self.redetect_frames = 0
        self.reacquisitions = 0
        self.id_switches = 0
        self.track_births = 0
        self.unique_ids = set()
        self.first_frame_ids = set()
        self.gaps: list[Gap] = []
        self.max_gap = 0
        self.total_gap_frames = 0
        self.det_total = 0
        self.det_frames = 0
        self.new_ids_total = 0
        self.events: list[dict] = []
        self._gap_start = None
        self._gap_kind = "coast"
        self._gap_id_before = None
        self._recent = deque(maxlen=int(self.window))
        self.ended_in_gap = False
        self._sustained_run = 0
        self.max_sustained_run = 0

    # ------------------------------------------------------------------ update
    def update(self, reading, dets, frame_index, t, redetect_ran=False, detect_ran=True):
        self.frames += 1
        ids = set()
        for d in dets:
            tid = getattr(d, "track_id", None)
            if tid is None:
                continue
            tid = int(tid)
            if tid not in self.unique_ids:
                self.unique_ids.add(tid)
                self.new_ids_total += 1
                if self.frames == 1:
                    self.first_frame_ids.add(tid)
                else:
                    self.track_births += 1
                    self.events.append({"frame": int(frame_index), "t": round(float(t), 3),
                                        "kind": "track_birth", "track_id": tid})
            ids.add(tid)
        self.det_total += len(dets)
        self.det_frames += 1
        if redetect_ran:
            self.redetect_calls += 1
            self.redetect_frames += 1
            self.events.append({"frame": int(frame_index), "t": round(float(t), 3),
                                "kind": "yolo_fallback", "track_id": reading.track_id,
                                "note": "low-confidence full-frame re-detect"})

        state = getattr(reading, "state", "")
        measured = bool(getattr(reading, "measured", False))
        if measured:
            self.sustained += 1
            self._recent.append(1)
            self._sustained_run += 1
            self.max_sustained_run = max(self.max_sustained_run, self._sustained_run)
            if self._gap_start is not None:
                length = frame_index - self._gap_start
                switch = (self._gap_id_before is not None and reading.track_id is not None
                          and reading.track_id != self._gap_id_before)
                self.gaps.append(Gap(self._gap_start, int(frame_index) - 1, length,
                                     kind=self._gap_kind,
                                     resumed_track_id=reading.track_id, id_switch=switch))
                self.max_gap = max(self.max_gap, length)
                self.reacquisitions += 1
                if switch:
                    self.id_switches += 1
                self.events.append({"frame": int(frame_index), "t": round(float(t), 3),
                                    "kind": "reacquire", "track_id": reading.track_id,
                                    "gap_frames": length,
                                    "id_switch": bool(switch)})
                self._gap_start = None
                self._gap_kind = "coast"
                self._gap_id_before = None
        else:
            self._sustained_run = 0
            self._recent.append(0)
            if state == "COASTING":
                self.coast += 1
                self._gap_kind = "coast"
            elif state == "LOST":
                self.lost += 1
                self._gap_kind = "lost"
            else:
                self.dead += 1
                self._gap_kind = "dead"
            self.total_gap_frames += 1
            if self._gap_start is None:
                self._gap_start = int(frame_index)
                self._gap_id_before = reading.track_id
        return self

    def close(self, frame_index):
        if self._gap_start is not None:
            length = frame_index - self._gap_start
            self.gaps.append(Gap(self._gap_start, int(frame_index), length,
                                 kind=self._gap_kind))
            self.max_gap = max(self.max_gap, length)
            self.ended_in_gap = True

    # ----------------------------------------------------------------- getters
    @property
    def fallback(self):
        return self.coast + self.lost + self.dead

    @property
    def sustained_pct(self):
        return 100.0 * self.sustained / max(self.frames, 1)

    @property
    def fallback_pct(self):
        return 100.0 * self.fallback / max(self.frames, 1)

    @property
    def recent_ratio(self):
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    def hud_lines(self):
        """Compact readout for the live window / HUD."""
        return [
            f"YOLO-confirmed {self.sustained} ({self.sustained_pct:.1f}%)  "
            f"fallback {self.fallback} ({self.fallback_pct:.1f}%)",
            f"  coast {self.coast}  lost {self.lost}  dead {self.dead}  "
            f"max gap {self.max_gap}f",
            f"  yolo re-detect calls {self.redetect_calls}  reacquired {self.reacquisitions}"
            + (f"  id switches {self.id_switches}" if self.id_switches else ""),
            f"  new track ids {self.track_births}  live ids {len(self.unique_ids)}  "
            f"det/frame {self.det_total / max(self.det_frames, 1):.1f}",
            f"  longest unbroken YOLO run {self.max_sustained_run}f"
            + (f"  window {self.recent_ratio * 100:.0f}% confirmed" if self._recent else ""),
        ]

    def summary(self):
        return {
            "frames": self.frames,
            "sustained_frames": self.sustained,
            "fallback_frames": self.fallback,
            "coast_frames": self.coast,
            "lost_frames": self.lost,
            "dead_frames": self.dead,
            "sustained_pct": round(self.sustained_pct, 2),
            "fallback_pct": round(self.fallback_pct, 2),
            "max_gap_frames": self.max_gap,
            "max_unbroken_confirmed_run": self.max_sustained_run,
            "total_gap_frames": self.total_gap_frames,
            "gaps": [g.as_dict() for g in self.gaps],
            "yolo_redetect_calls": self.redetect_calls,
            "reacquisitions": self.reacquisitions,
            "id_switches": self.id_switches,
            "new_tracks_after_first_frame": self.track_births,
            "unique_track_ids": sorted(self.unique_ids),
            "detections_per_frame": round(self.det_total / max(self.det_frames, 1), 2),
            "verdict": self.verdict(),
            "events": self.events[:500],
        }

    def verdict(self):
        """Plain-language read on the counters. A gap that was recovered is normal
        behaviour (occlusion); a gap that was not recovered is the problem."""
        if self.frames == 0:
            return "no frames processed"
        pct = self.fallback_pct
        recovered = self.reacquisitions >= len(self.gaps) - (1 if self.ended_in_gap else 0)
        if pct < 2.0:
            return ("tracker sustained the subject almost entirely on detections; "
                    "BoT-SORT rarely needed a re-detect")
        if recovered and not self.ended_in_gap:
            return (f"tracker fell back to its motion model on {pct:.1f}% of frames "
                    f"({self.coast} coast, {self.lost} lost) across {len(self.gaps)} gap(s) "
                    f"and recovered every one; max gap {self.max_gap} frames, "
                    f"{self.id_switches} ID switch(es)")
        if self.ended_in_gap:
            return (f"the run ended while the subject was still missing "
                    f"({self.fallback} fallback frames so far, max gap {self.max_gap} frames)")
        return (f"tracker lost the subject and did not recover it "
                f"({self.fallback} fallback frames, {self.lost} lost, "
                f"max gap {self.max_gap} frames, {self.id_switches} ID switches): "
                f"lower conf, raise track_buffer, or loosen the appearance gates")

    def print_report(self, prefix="[vt] "):
        lines = self.hud_lines()
        print(prefix + "fallback accounting")
        for ln in lines:
            print(prefix + ln.strip() if ln.startswith(" ") else prefix + ln)
        print(prefix + "verdict: " + self.verdict())
