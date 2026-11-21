"""Source -> detected, tracked, synced subject.

This module owns the per-frame pipeline and the offline (batch) runner. The live
runner in vt/live.py drives the exact same `Pipeline.step_frame`, so what you see
in the preview window and what gets written to disk cannot diverge.

Source can be a video file, a camera index ("0"), or a stream URL
(rtsp://, rtmp://, http://). Live sources are what the live runner is for.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .config import load_config
from .detect import build_detector
from .face import FaceProbe
from .flow import LKRefiner
from .geometry import focal_px_from_hfov, frame_diagonal, iou_xyxy
from .gmc import GlobalMotion
from .reid import make_embedder
from .render import Renderer
from .select import resolve_selection
from .stats import FallbackStats
from .sync import LatencyModel, TrajectoryBuffer
from .tracker import (STATE_COASTING, STATE_DEAD, STATE_LOCKED, STATE_LOST,
                      IdAssigner, SubjectTracker)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://",
                   "v4l2://", "gst://")


def source_is_live(source):
    s = str(source)
    if s.isdigit():
        return True
    return s.lower().startswith(STREAM_PREFIXES)


class VideoSink:
    """Annotated output. Prefers an ffmpeg h264 pipe (plays anywhere, previews in
    chat apps); falls back to cv2.VideoWriter."""

    def __init__(self, path, w, h, fps, codec="mp4v", h264=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = None
        self.writer = None
        self.mode = "none"
        if h264 and shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.mode = "ffmpeg-h264"
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, (w, h))
            self.mode = f"cv2-{codec}"

    def write(self, frame):
        if self.proc is not None:
            if frame.shape[1] % 2 or frame.shape[0] % 2:
                frame = frame[:frame.shape[0] - frame.shape[0] % 2,
                              :frame.shape[1] - frame.shape[1] % 2]
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        elif self.writer is not None:
            self.writer.write(frame)

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.wait()
        if self.writer is not None:
            self.writer.release()


class Pipeline:
    def __init__(self, cfg, source):
        self.cfg = cfg
        self.source = source
        self.is_live = source_is_live(source)
        self.stem = Path(str(source)).stem if not self.is_live else f"live_{source}"
        if self.is_live and str(source).isdigit():
            self.stem = f"cam{source}"
        self.out_dir = Path(cfg["output"]["dir"]).expanduser()
        if not self.out_dir.is_absolute():
            self.out_dir = Path.cwd() / self.out_dir
        self.sink = None
        self.det = None
        self.embedder = None
        self.face_probe = None
        self.gmc = None
        self.assigner = None
        self.tracker = None
        self.renderer = None
        self.traj = TrajectoryBuffer(maxlen=int(cfg["tracker"].get("trail_len", 180)) * 4)
        self.latency = LatencyModel(alpha=0.2,
                                    fixed_ms=cfg["sync"].get("fixed_latency_ms", 0.0))
        self.stats = FallbackStats()
        self.use_botsort = False
        self.summary_stats = {}
        self.n_processed = 0
        self.det_ms_ema = 0.0
        self.loop_ms_ema = 0.0
        self.redetect_calls = 0
        self.detect_every = max(int(cfg["detector"].get("every_n", 1)), 1)
        self._frame_dets = []
        self._last_frame = None
        self._processing_ms = 0.0

    # ------------------------------------------------------------------ outputs
    def _out_path(self, key, default_suffix, ext):
        explicit = self.cfg["output"].get(key)
        if explicit:
            p = Path(explicit).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir / f"{self.stem}{default_suffix}{ext}"

    # ------------------------------------------------------------------- source
    def open_source(self):
        src = self.source
        if str(src).isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source: {self.source!r}")
        if not self.is_live:
            p = Path(str(self.source)).expanduser()
            if not p.exists():
                raise SystemExit(f"video not found: {p}")
        fps = float(self.cfg["video"].get("fps_override") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0 or not np.isfinite(fps) or fps > 480:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = 0 if self.is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if w <= 0 or h <= 0:
            ok, probe = cap.read()
            if not ok:
                raise SystemExit(f"source opened but produced no frames: {self.source!r}")
            h, w = probe.shape[:2]
            self._last_frame = probe
        return cap, fps, w, h, total

    def describe(self):
        cfg = self.cfg
        print(f"[vt] source     {self.source}  {self.src_w}x{self.src_h} @ {self.src_fps:.2f} fps  "
              f"{self.src_total if self.src_total > 0 else 'live/unknown'} frames"
              f"{'  (LIVE)' if self.is_live else ''}")
        print(f"[vt] backend    {self.backend}  model={cfg['detector']['model']} "
              f"imgsz={cfg['detector']['imgsz']} conf={cfg['detector']['conf']}"
              f"{'' if self.detect_every == 1 else f'  every {self.detect_every} frames'}")
        print(f"[vt] tracker    "
              f"{'ultralytics ' + str(cfg['tracker']['name']) if self.use_botsort else 'own association'}"
              f"  | reid={self.embedder.name} | "
              f"gmc={'inside botsort' if (not self.own_gmc and self.use_botsort) else self.own_gmc}")

    # -------------------------------------------------------------------- build
    def build(self, h, w):
        cfg = self.cfg
        dc = dict(cfg["detector"])
        backend = str(dc.get("backend", "auto")).lower()
        if backend == "auto":
            try:
                import ultralytics  # noqa: F401
                backend = "ultralytics"
            except Exception:
                backend = "onnx"
        tname = str(cfg["tracker"].get("name", "botsort")).lower()
        if backend == "onnx":
            mp = Path(str(dc.get("model", ""))).expanduser()
            if not mp.exists():
                raise SystemExit(
                    f"onnx backend needs a local model file: '{mp}' not found.\n"
                    "  Either install the ultralytics backend (pip install ultralytics) so\n"
                    "  yolov8n.pt can be used/downloaded, or export an INT8 model first:\n"
                    "    python tools/export_int8.py --model yolov8n.pt --to onnx-int8 \\\n"
                    "        --calib-video YOURCLIP.mp4 --imgsz 320 --out models/")
        if backend == "dummy":
            gt = dc.get("dummy_gt")
            if not gt or not Path(str(gt)).expanduser().exists():
                raise SystemExit(f"dummy backend needs --dummy-gt pointing at an existing "
                                 f"json file (got {gt!r})")
        self.use_botsort = backend == "ultralytics" and tname in ("botsort", "bytetrack")
        if self.use_botsort:
            tcfg = cfg["tracker"].get("config")
            if tname == "bytetrack":
                tcfg = None
            dc["tracker_config"] = str(Path(__file__).resolve().parent.parent / tcfg) \
                if tcfg else None
        self.det = build_detector(dc, track=self.use_botsort)
        self.backend = backend
        self.embedder = make_embedder(cfg["reid"])
        fc = cfg["face"]
        self.face_probe = FaceProbe(enable=fc.get("enable", True),
                                    scale_factor=fc.get("scale_factor", 1.1),
                                    min_neighbors=fc.get("min_neighbors", 5),
                                    every_n=fc.get("every_n", 1)) if fc.get("enable", True) else None
        focal = float(cfg["camera"].get("focal_px") or 0.0)
        if focal <= 0:
            focal = focal_px_from_hfov(w, cfg["camera"].get("hfov_deg", 65.0))
        self.camera = {"focal_px": focal, "ipd_m": cfg["camera"].get("ipd_m", 0.063),
                       "source": "explicit" if cfg["camera"].get("focal_px") else "hfov"}
        own_gmc = bool(cfg["tracker"].get("gmc", True))
        if self.use_botsort and own_gmc:
            own_gmc = False  # BoT-SORT already runs its own GMC; do not double-compensate
        self.gmc = GlobalMotion(enable=own_gmc, max_shift_px=0.25 * frame_diagonal((h, w)))
        self.own_gmc = own_gmc
        if bool(cfg["tracker"].get("own_ids", True)) and not self.use_botsort:
            self.assigner = IdAssigner(iou_min=cfg["tracker"].get("id_iou_min", 0.3),
                                       max_age=int(cfg["tracker"].get("coast_max_frames", 30)))
        self.renderer = Renderer(cfg["output"], names=getattr(self.det, "names", {}))
        return backend

    def _redetect(self, frame, frame_index):
        """Full-frame low-confidence re-detect used once the lock is LOST.
        Reuses the same model instance (no second copy resident on the device)."""
        low = float(self.cfg["detector"].get("redetect_conf", 0.15))
        if not hasattr(self.det, "conf"):
            got = self.det.detect(frame, frame_index)  # backend ignores conf
        else:
            if low >= float(self.det.conf):
                return []
            old = self.det.conf
            self.det.conf = low
            try:
                got = self.det.detect(frame, frame_index)
            finally:
                self.det.conf = old
        self.redetect_calls += 1
        kept = []
        for d in got:
            if all(iou_xyxy(d.xyxy, x.xyxy) < 0.6 for x in self._frame_dets):
                kept.append(d)
        return kept

    # --------------------------------------------------------------- selection
    def select(self, cap, fps):
        cfg = self.cfg["selection"]
        png = self._out_path("selection_png", "_selection", ".png") \
            if self.cfg["output"].get("save_selection_png", True) else None
        last_detail = {}
        if self.is_live:
            # a camera cannot seek: pull frames until something is detectable
            max_tries = max(int(float(cfg.get("scan_sec", 2.0)) * fps), 15)
            tries = list(range(max_tries))
        else:
            start_f = int(round(float(cfg.get("at_sec", 0.0)) * fps))
            tries = [start_f]
            k = 0.5
            while k <= float(cfg.get("scan_sec", 2.0)):
                tries.append(start_f + int(round(k * fps)))
                k += 0.5
        for i, f in enumerate(tries):
            if self._last_frame is not None:
                frame = self._last_frame
                self._last_frame = None
            else:
                if not self.is_live:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
            dets = self.det.detect(frame, f)
            dets = [d for d in dets if d.score >= float(self.cfg["detector"].get("conf", 0.25))]
            if not dets:
                last_detail = {"reason": f"no detections at frame {f}"}
                continue
            idx, detail = resolve_selection(cfg, frame, dets,
                                            save_path=str(png) if png else None)
            detail["frame_index"] = f
            detail["n_det"] = len(dets)
            if idx is not None:
                return idx, dets[idx], frame, detail
            last_detail = detail
        return None, None, None, last_detail

    # ------------------------------------------------------------------ tracker
    def init_tracker(self, sel_det, sel_frame, detail, fps):
        cfg = self.cfg
        flow = LKRefiner(enable=bool(cfg["tracker"].get("flow_refine", False)))
        self.tracker = SubjectTracker(
            tracker_cfg=cfg["tracker"], reid_cfg=cfg["reid"], depth_cfg=cfg["depth"],
            face_cfg=cfg["face"], embedder=self.embedder,
            frame=sel_frame, bbox=sel_det.xyxy,
            t=float(detail.get("frame_index", 0)) / fps,
            frame_index=int(detail.get("frame_index", 0)),
            track_id=sel_det.track_id, score=sel_det.score, cls_id=sel_det.cls_id,
            face_probe=self.face_probe, camera=self.camera,
            flow_refiner=flow if flow.enable else None)
        return self.tracker

    # ---------------------------------------------------------------- one frame
    def step_frame(self, frame, frame_index, t_frame, fps=30.0, force_detect=True,
                   extra_latency_ms=0.0):
        """One frame of work. Shared by the batch runner and the live runner."""
        cfg = self.cfg
        t0 = time.perf_counter()
        run_detect = force_detect and (frame_index % self.detect_every == 0)

        M = self.gmc.update(frame) if self.gmc.enable else None
        det_ms = 0.0
        if run_detect:
            d0 = time.perf_counter()
            dets = self.det.detect_track(frame, frame_index)
            det_ms = (time.perf_counter() - d0) * 1000.0
            if self.assigner is not None:
                dets = self.assigner.assign(dets)
        else:
            dets = []
        self.det_ms_ema = det_ms if self.n_processed == 0 else \
            0.9 * self.det_ms_ema + 0.1 * det_ms
        self._frame_dets = dets

        red_before = self.redetect_calls
        reading = self.tracker.step(frame, dets, t_frame, frame_index, M=M,
                                    redetect=self._redetect)
        redetect_ran = self.redetect_calls > red_before

        self.traj.push(t_frame, [reading.center[0], reading.center[1], reading.depth_m or 0.0],
                       [reading.velocity[0], reading.velocity[1], 0.0],
                       state=reading.state, bbox=reading.bbox)

        loop_ms = (time.perf_counter() - t0) * 1000.0
        self.loop_ms_ema = loop_ms if self.n_processed == 0 else \
            0.9 * self.loop_ms_ema + 0.1 * loop_ms

        pred = None
        if cfg["sync"].get("latency_comp", True):
            pred = self.traj.sample(t_frame + self.latency.s,
                                    max_extrapolate_s=cfg["sync"].get(
                                        "max_extrapolate_ms", 120.0) / 1000.0)
        # Measured end-to-end latency: perception work plus whatever the caller
        # spent on presentation last frame. Updated after `pred` so the sample
        # above uses the previous estimate, not a value derived from itself.
        self.latency.update(loop_ms + float(extra_latency_ms))

        info = {
            "frame_index": frame_index, "t": t_frame, "n_det": len(dets),
            "det_ms": det_ms, "loop_ms": loop_ms, "latency_ms": self.latency.ms,
            "fps": 1000.0 / max(self.loop_ms_ema, 1e-6),
            "detected": run_detect,
            "gmc": (f"{self.gmc.last_info.get('inliers', 0)}in"
                    if (self.own_gmc and self.gmc.last_info.get("ok"))
                    else ("off" if not self.own_gmc else "n/a")),
            "pred_t": (pred["t"] if pred else None),
            "pred_p": (pred["p"][:2] if pred else None),
            "stats": self.stats,
        }
        rec = reading.as_dict()
        rec.update({
            "detected": run_detect,
            "n_det": len(dets),
            "det_ms": round(det_ms, 3),
            "loop_ms": round(loop_ms, 3),
            "latency_ms": round(self.latency.ms, 3),
            "target_now": ([float(pred["p"][0]), float(pred["p"][1])] if pred else None),
            "target_now_t": (float(pred["t"]) if pred else None),
            "extrapolated": (bool(pred["extrapolated"]) if pred else None),
            "detections": [[float(v) for v in d.xyxy] + [float(d.score), int(d.cls_id),
                                                          (int(d.track_id) if d.track_id is not None else -1)]
                           for d in dets],
        })
        self.n_processed += 1
        return {"reading": reading, "dets": dets, "info": info, "rec": rec,
                "det_ms": det_ms, "loop_ms": loop_ms, "pred": pred,
                "redetect_ran": redetect_ran, "detected": run_detect}

    def account(self, res, frame_index, t_frame):
        """Feed one result into the fallback counters (call after step_frame)."""
        self.stats.update(res["reading"], res["dets"], frame_index, t_frame,
                          redetect_ran=res["redetect_ran"],
                          detect_ran=res["detected"])

    def annotate(self, frame, res, extra_lines=None):
        info = dict(res["info"])
        if extra_lines:
            info["extra"] = list(extra_lines)
        return self.renderer.draw(frame, res["dets"], res["reading"],
                                  self.tracker.trail, info)
    # --------------------------------------------------------------------- main
    def run(self):
        cfg = self.cfg
        cap, fps, w, h, total = self.open_source()
        self.src_fps, self.src_w, self.src_h, self.src_total = fps, w, h, total
        self.build(h, w)
        self.describe()
        print(f"[vt] camera     focal={self.camera['focal_px']:.1f}px ({self.camera['source']}) "
              f"ipd={self.camera['ipd_m']}m | depth={cfg['depth']['mode']} | "
              f"face={bool(self.face_probe and self.face_probe.enable)}")

        idx, sel_det, sel_frame, detail = self.select(cap, fps)
        if idx is None or sel_det is None:
            print(f"[vt] no subject selected: {detail.get('reason', 'aborted')}")
            return {"ok": False, "reason": detail.get("reason", "no selection"), "detail": detail}
        print(f"[vt] subject    [{idx}] {sel_det.cls_name} conf={sel_det.score:.2f} "
              f"bbox={[round(float(v)) for v in sel_det.xyxy]} frame={detail.get('frame_index')}"
              + (f"  preview={detail.get('frame_png')}" if detail.get("frame_png") else ""))
        self.init_tracker(sel_det, sel_frame, detail, fps)

        out_video = self._out_path("video", "_tracked", ".mp4")
        out_jsonl = self._out_path("jsonl", "_track", ".jsonl")
        self.sink = VideoSink(out_video, w, h, fps, codec=cfg["output"].get("codec", "mp4v"),
                              h264=bool(cfg["output"].get("h264", True)))
        print(f"[vt] writing    {out_video}  ({self.sink.mode})")
        print(f"[vt] records    {out_jsonl}")

        start_frame = int(cfg["video"].get("start_frame", 0))
        stride = max(int(cfg["video"].get("stride", 1)), 1)
        max_frames = int(cfg["video"].get("max_frames", 0))
        if not self.is_live and start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fh = open(out_jsonl, "w")
        frame_index = start_frame
        t_wall0 = time.perf_counter()
        state_counts = {STATE_LOCKED: 0, STATE_COASTING: 0, STATE_LOST: 0, STATE_DEAD: 0}
        try:
            while True:
                if stride > 1:
                    for _ in range(stride - 1):
                        cap.grab()
                ok, frame = cap.read()
                if not ok:
                    break
                t_frame = frame_index / fps
                res = self.step_frame(frame, frame_index, t_frame, fps)
                self.account(res, frame_index, t_frame)
                reading = res["reading"]
                state_counts[reading.state] = state_counts.get(reading.state, 0) + 1
                self.annotate(frame, res, extra_lines=self.stats.hud_lines())
                self.sink.write(frame)
                fh.write(json.dumps(res["rec"]) + "\n")
                if self.n_processed % int(cfg["output"].get("progress_every", 25)) == 0:
                    print(f"  {self.n_processed:5d} frames  t={t_frame:6.2f}s  "
                          f"{reading.state:8s} miss={reading.misses:3d} sim={reading.sim:.2f} "
                          f"det={self.det_ms_ema:5.1f}ms loop={self.loop_ms_ema:5.1f}ms  "
                          f"confirmed={self.stats.sustained} fallback={self.stats.fallback}",
                          flush=True)
                _ = res  # keep the reference for debuggers
                frame_index += stride
                if max_frames and self.n_processed >= max_frames:
                    break
        except KeyboardInterrupt:
            print("\n[vt] interrupted, writing what we have")
        finally:
            fh.close()
            self.sink.close()
            cap.release()
            self.stats.close(frame_index)

        self.stats.print_report("[vt] ")
        wall = time.perf_counter() - t_wall0
        summary = {
            "ok": True,
            "source": str(self.source),
            "frames": self.n_processed,
            "fps_source": fps,
            "backend": self.backend,
            "model": str(cfg["detector"]["model"]),
            "imgsz": cfg["detector"]["imgsz"],
            "detect_every": self.detect_every,
            "tracker": cfg["tracker"]["name"] if self.use_botsort else "own-association",
            "embedder": self.embedder.name,
            "gmc": "botsort" if (self.use_botsort and not self.own_gmc) else
                   ("own" if self.own_gmc else "off"),
            "selection": {"index": int(idx), "cls": sel_det.cls_name,
                          "bbox": [float(v) for v in sel_det.xyxy],
                          "frame_index": detail.get("frame_index")},
            "state_frames": {k: int(v) for k, v in state_counts.items()},
            "fallback": self.stats.summary(),
            "tracker_stats": self.tracker.summary(),
            "avg_det_ms": round(self.det_ms_ema, 3),
            "avg_loop_ms": round(self.loop_ms_ema, 3),
            "end_to_end_fps": round(self.n_processed / wall, 2) if wall > 0 else None,
            "avg_latency_ms": round(self.latency.ms, 3),
            "depth_source": self.camera["source"],
            "focal_px": round(self.camera["focal_px"], 2),
            "outputs": {"video": str(out_video), "jsonl": str(out_jsonl)},
        }
        sp = out_jsonl.with_suffix(".summary.json")
        with open(sp, "w") as f2:
            json.dump(summary, f2, indent=2)
        summary["outputs"]["summary"] = str(sp)
        self.summary_stats = summary
        return summary


def run(cfg=None, source=None):
    cfg = cfg or load_config()
    return Pipeline(cfg, source).run()
