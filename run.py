#!/usr/bin/env python3
"""Edge vision tracker CLI.

Two modes:

  live     watch the tracking while the footage plays, in real time (window,
           keyboard control, running fallback counters). Use --play.
  batch    process a file as fast as possible and write an annotated mp4 plus a
           per-frame record for tuning. Default.

Examples
--------
  # LIVE: preview window, click the object, watch the boxes update in real time
  python run.py yourclip.mp4 --play

  # LIVE camera (index 0) or a stream, biggest person picked automatically
  python run.py 0 --play --no-gui
  python run.py 'rtsp://<user>:<password>@<camera-ip>:554/stream' --play --select point --point 640,360

  # how long can BoT-SORT hold the subject without calling YOLO? detect every
  # 3rd frame and read the fallback accounting at the end
  python run.py yourclip.mp4 --play --detect-every 3

  # BATCH: annotated mp4 + jsonl for offline analysis
  python run.py yourclip.mp4

  # INT8 ONNX on the device, no torch
  python run.py yourclip.mp4 --backend onnx --model models/yolov8n_int8.onnx --imgsz 320
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vt.config import load_config, set_by_path  # noqa: E402


def parse_point(s):
    x, y = [float(v) for v in str(s).replace(" ", "").split(",")[:2]]
    return [x, y]


def parse_roi(s):
    vals = [float(v) for v in str(s).replace(" ", "").split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("roi must be x,y,w,h")
    return vals


def build_parser():
    p = argparse.ArgumentParser(description="YOLO + BoT-SORT primary-subject tracker",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?",
                   help="video file, camera index (0), or stream URL (rtsp://...)")
    p.add_argument("--source", dest="source_flag", default=None,
                   help="same as the positional source; takes precedence if both given")
    p.add_argument("--play", "--live", dest="live", action="store_true",
                   help="live mode: real-time preview window with running fallback counters")
    p.add_argument("--speed", type=float, default=None,
                   help="live playback speed, 1.0 = real time, 0 = as fast as possible")
    p.add_argument("--no-window", dest="no_window", action="store_true", default=None,
                   help="live mode without a preview window (headless box, logging only)")
    p.add_argument("--live-window", default=None, help="preview window title")
    p.add_argument("--record", default=None,
                   help="live mode: also write the annotated preview to this mp4")
    p.add_argument("--stats", default=None, help="where to write the fallback stats json")
    p.add_argument("--quiet", action="store_true", default=None,
                   help="live mode: do not print progress lines")
    p.add_argument("--points", action="store_true", default=None,
                   help="draw the trail as tracking points instead of a line")
    p.add_argument("--keypoints", action="store_true", default=None,
                   help="draw pose keypoints (needs a -pose model, e.g. yolov8n-pose.pt)")
    p.add_argument("--subject-only", dest="subject_only", action="store_true", default=None,
                   help="draw only the tracked subject, not every detection")

    p.add_argument("--config", default=None, help="yaml config (default: ./config.yaml)")
    p.add_argument("--profile", default=None, choices=["balanced", "strict"])
    p.add_argument("--out", default=None, help="annotated output video path (batch mode)")
    p.add_argument("--jsonl", default=None, help="per-frame record path")
    p.add_argument("--out-dir", default=None, help="output directory")

    p.add_argument("--model", default=None,
                   help="yolov8n.pt | yolov8n-pose.pt | yolov8n_int8.onnx | _int8.tflite")
    p.add_argument("--backend", default=None, choices=["auto", "ultralytics", "onnx", "dummy"])
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--classes", default=None,
                   help="comma separated class ids, e.g. 0 for person; empty string = all")
    p.add_argument("--device", default=None, help="cpu | cuda:0 | mps")
    p.add_argument("--half", action="store_true", default=None)
    p.add_argument("--onnx-format", default=None, choices=["auto", "v8", "v8_t", "v5", "e2e"])
    p.add_argument("--redetect-conf", type=float, default=None,
                   help="low-confidence threshold used for LOST re-detection")
    p.add_argument("--detect-every", type=int, default=None,
                   help="run YOLO every Nth frame and let the tracker carry the rest. "
                        "1 = every frame. This measures how long BoT-SORT holds alone.")
    p.add_argument("--dummy-gt", default=None, help="ground-truth json for backend=dummy")

    p.add_argument("--tracker", default=None, choices=["botsort", "bytetrack", "none"])
    p.add_argument("--tracker-config", default=None, help="tracker yaml (default configs/botsort_reid.yaml)")
    p.add_argument("--coast-max", type=int, default=None, help="frames before switching to LOST")
    p.add_argument("--lost-max", type=int, default=None, help="frames before giving up")
    p.add_argument("--no-gmc", action="store_true", default=None)
    p.add_argument("--flow-refine", action="store_true", default=None)

    p.add_argument("--embedder", default=None, choices=["lite", "mobilenet", "none"])
    p.add_argument("--reid-weights", default=None, help="onnx weights for embedder=mobilenet")
    p.add_argument("--ema-alpha", type=float, default=None)
    p.add_argument("--score-gate", type=float, default=None, help="template update confidence gate")
    p.add_argument("--drift-floor", type=float, default=None)

    p.add_argument("--select", default=None, choices=["click", "auto", "point", "roi"])
    p.add_argument("--select-at", type=float, default=None, help="seconds into the clip")
    p.add_argument("--point", type=parse_point, default=None, help="x,y")
    p.add_argument("--roi", type=parse_roi, default=None, help="x,y,w,h")
    p.add_argument("--auto-rule", default=None, choices=["largest", "conf"])
    p.add_argument("--no-gui", dest="no_gui", action="store_true", default=None,
                   help="never open a selection window; use --select auto/point/roi")

    p.add_argument("--depth", default=None, choices=["ipd", "pseudo", "off"])
    p.add_argument("--no-face", action="store_true", default=None)
    p.add_argument("--hfov", type=float, default=None, help="horizontal FOV degrees")
    p.add_argument("--focal-px", type=float, default=None, help="calibrated focal length in px")
    p.add_argument("--ipd-m", type=float, default=None, help="real inter-pupillary distance (m)")

    p.add_argument("--stride", type=int, default=None,
                   help="skip source frames entirely (different from --detect-every)")
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps-override", type=float, default=None)
    p.add_argument("--motor-hz", type=float, default=None)
    p.add_argument("--latency-ms", type=float, default=None, help="fixed extra latency to compensate")
    p.add_argument("--no-latency-comp", action="store_true", default=None)
    p.add_argument("--no-h264", action="store_true", default=None)
    p.add_argument("--print-config", action="store_true")
    return p


def overrides_from_args(a):
    o = {}

    def put(path, val):
        if val is not None:
            set_by_path(o, path, val)

    put("profile", a.profile)
    put("output.video", a.record or (a.out if not a.live else None))
    put("output.jsonl", a.jsonl)
    put("output.dir", a.out_dir)
    put("output.stats", a.stats)
    if a.points:
        put("output.points", True)
    if a.keypoints:
        put("output.keypoints", True)
    if a.subject_only:
        put("output.all_tracks", False)
    put("live.speed", a.speed)
    put("live.window", a.live_window)
    put("live.quiet", a.quiet)
    if a.no_window:
        put("live.window_enabled", False)
    put("detector.model", a.model)
    put("detector.backend", a.backend)
    put("detector.imgsz", a.imgsz)
    put("detector.conf", a.conf)
    put("detector.iou", a.iou)
    put("detector.device", a.device)
    put("detector.half", a.half)
    put("detector.onnx_format", a.onnx_format)
    put("detector.redetect_conf", a.redetect_conf)
    put("detector.every_n", a.detect_every)
    put("detector.dummy_gt", a.dummy_gt)
    if a.classes is not None:
        o.setdefault("detector", {})["classes"] = [int(v) for v in a.classes.split(",")
                                                   if v.strip() != ""]
    put("tracker.name", a.tracker)
    put("tracker.config", a.tracker_config)
    put("tracker.coast_max_frames", a.coast_max)
    put("tracker.lost_max_frames", a.lost_max)
    if a.no_gmc:
        put("tracker.gmc", False)
    if a.flow_refine:
        put("tracker.flow_refine", True)
    put("reid.embedder", a.embedder)
    put("reid.weights", a.reid_weights)
    put("reid.ema_alpha", a.ema_alpha)
    put("reid.update_score_gate", a.score_gate)
    put("reid.drift_floor", a.drift_floor)
    put("selection.mode", a.select)
    put("selection.at_sec", a.select_at)
    put("selection.point", a.point)
    put("selection.roi", a.roi)
    put("selection.auto_rule", a.auto_rule)
    if a.no_gui and not (o.get("selection", {}) or {}).get("mode"):
        put("selection.mode", "auto")
    put("depth.mode", a.depth)
    if a.no_face:
        put("face.enable", False)
        put("depth.mode", a.depth or "off")
    put("camera.hfov_deg", a.hfov)
    put("camera.focal_px", a.focal_px)
    put("camera.ipd_m", a.ipd_m)
    put("video.stride", a.stride)
    put("video.start_frame", a.start_frame)
    put("video.max_frames", a.max_frames)
    put("video.fps_override", a.fps_override)
    put("sync.motor_hz", a.motor_hz)
    put("sync.fixed_latency_ms", a.latency_ms)
    if a.no_latency_comp:
        put("sync.latency_comp", False)
    if a.no_h264:
        put("output.h264", False)
    return o


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        if not (args.source or args.source_flag):
            return 0
    source = args.source_flag or args.source
    if not source:
        build_parser().error("a source is required: a video path, a camera index "
                             "(0), or a stream URL (or use --print-config alone)")

    if args.live:
        from vt.live import LiveRunner
        summary = LiveRunner(cfg, source).run()
    else:
        from vt.pipeline import Pipeline
        summary = Pipeline(cfg, source).run()
        print("\n[vt] summary")
        for k in ("frames", "backend", "model", "imgsz", "detect_every", "tracker",
                  "embedder", "avg_det_ms", "avg_loop_ms", "end_to_end_fps",
                  "avg_latency_ms", "focal_px"):
            if k in summary:
                print(f"  {k:16s} {summary[k]}")
        if summary.get("fallback"):
            fb = summary["fallback"]
            print(f"  {'confirmed':16s} {fb['sustained_frames']} frames "
                  f"({fb['sustained_pct']}%)")
            print(f"  {'fallback':16s} {fb['fallback_frames']} frames "
                  f"({fb['fallback_pct']}%)  coast {fb['coast_frames']}  "
                  f"lost {fb['lost_frames']}  max gap {fb['max_gap_frames']}f")
            print(f"  {'yolo re-detect':16s} {fb['yolo_redetect_calls']} calls   "
                  f"reacquired {fb['reacquisitions']}   id switches {fb['id_switches']}")
            print(f"  {'verdict':16s} {fb['verdict']}")
        if summary.get("outputs"):
            for k, v in summary["outputs"].items():
                print(f"  {k:16s} {v}")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
