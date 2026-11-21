"""Live runner: watch the tracking while the footage plays, in real time.

This is the mode the tracker is meant to run in. It drives the same
`Pipeline.step_frame` as the batch runner, so the box you see on screen is the
box that would be handed to the motor, with the same latency compensation.

Source can be a video file (played back at its own frame rate, so you can see
exactly how the tracker behaves in real time), a camera index ("0"), or a stream
URL (rtsp://..., http://...).

Keys, when the window has focus:
    space   pause / resume
    s       step one frame (pauses first)
    r       reset the fallback counters
    h       toggle the HUD
    p       toggle trail style: line <-> tracking points
    d       toggle drawing every detection, not just the subject
    + / -   playback speed (files only)
    q, esc  quit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .pipeline import Pipeline, VideoSink
from .stats import FallbackStats


def _sleep_until(deadline, chunk=0.002):
    """Sleep until `deadline`. time.sleep() overshoots badly on some hosts (a 16 ms
    request can sleep 130 ms on a low-power Intel laptop), so pace in small chunks
    instead of trusting one long sleep."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(chunk, remaining))


class LiveRunner:
    def __init__(self, cfg, source):
        self.cfg = cfg
        self.source = source
        self.pipe = Pipeline(cfg, source)

    # ------------------------------------------------------------------ helpers
    def _hud_extra(self, res, speed, paused, recording, windowed):
        p = self.pipe
        lines = list(p.stats.hud_lines())
        mode = ("LIVE" if p.is_live else f"FILE x{speed:g}")
        if paused:
            mode += "  PAUSED"
        if recording:
            mode += "  REC"
        lines.append(f"  {mode}  detect every {p.detect_every}f  "
                     f"{'window' if windowed else 'no window'}"
                     + ("   [space]pause [s]tep [r]eset [p]oints [q]uit" if windowed else ""))
        return lines

    def _key(self, key, speed, paused):
        """Returns (speed, paused, step_once, quit)."""
        step = False
        quit_ = False
        if key == -1:
            return speed, paused, step, quit_
        c = chr(key) if 32 <= key < 127 else ""
        if c in ("q",) or key == 27:
            quit_ = True
        elif c == " ":
            paused = not paused
        elif c == "s":
            paused = True
            step = True
        elif c == "r":
            self.pipe.stats = FallbackStats()
            print("[live] fallback counters reset")
        elif c == "h":
            self.pipe.renderer.hud = not self.pipe.renderer.hud
        elif c == "p":
            self.pipe.renderer.points = not self.pipe.renderer.points
        elif c == "d":
            self.pipe.renderer.all_tracks = not self.pipe.renderer.all_tracks
        elif c in ("+", "="):
            speed = min(speed * 2.0, 16.0) if speed > 0 else 1.0
        elif c == "-":
            speed = max(speed / 2.0, 0.25)
        elif c == "0":
            speed = 0.0
        return speed, paused, step, quit_

    # --------------------------------------------------------------------- main
    def run(self):
        cfg = self.cfg
        lc = cfg["live"]
        p = self.pipe
        cap, fps, w, h, total = p.open_source()
        p.src_fps, p.src_w, p.src_h, p.src_total = fps, w, h, total
        p.build(h, w)
        p.describe()
        print(f"[vt] live       speed x{lc.get('speed', 1.0)}"
              f"{'  (0 = as fast as possible)' if float(lc.get('speed', 1.0)) == 0 else ''}"
              f"  catch_up={bool(lc.get('catch_up', True))}")
        print("[vt] window     click the object to track, then press Enter")

        idx, sel_det, sel_frame, detail = p.select(cap, fps)
        if idx is None or sel_det is None:
            print(f"[live] no subject selected: {detail.get('reason', 'aborted')}")
            return {"ok": False, "reason": detail.get("reason", "no selection")}
        print(f"[live] subject  [{idx}] {sel_det.cls_name} conf={sel_det.score:.2f} "
              f"bbox={[round(float(v)) for v in sel_det.xyxy]} "
              f"frame={detail.get('frame_index')}")
        p.init_tracker(sel_det, sel_frame, detail, fps)

        window = str(lc.get("window", "vision tracker"))
        show = bool(lc.get("window_enabled", True)) and cv2 is not None
        if show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(w, 1280), min(h, 800))

        recording = bool(cfg["output"].get("video"))
        sink = VideoSink(cfg["output"]["video"], w, h, fps,
                         codec=cfg["output"].get("codec", "mp4v"),
                         h264=bool(cfg["output"].get("h264", True))) if recording else None
        if sink:
            print(f"[live] recording {cfg['output']['video']}  ({sink.mode})")
        rec_path = cfg["output"].get("jsonl")
        fh = open(rec_path, "w") if rec_path else None
        if fh:
            print(f"[live] records   {rec_path}")

        speed = float(lc.get("speed", 1.0))
        catch_up = bool(lc.get("catch_up", True))
        max_skip = int(lc.get("max_skip", 30))
        max_frames = int(cfg["video"].get("max_frames", 0))
        progress_every = int(lc.get("progress_every", 60))

        pending = sel_frame if lc.get("reuse_selection_frame", True) else None
        frame_index = int(detail.get("frame_index", 0)) + (0 if pending is not None else 1)
        paused = False
        step_once = False
        render_ms = 0.0
        t0 = time.perf_counter()
        dropped = 0
        last_report = 0
        quiet = bool(lc.get("quiet", False))

        try:
            while True:
                if not paused or step_once:
                    if pending is not None:
                        frame = pending
                        pending = None
                    else:
                        ok = False
                        skips = 0
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            if p.is_live or not catch_up or speed <= 0:
                                break
                            target = t0 + (frame_index / fps) / speed
                            behind = time.perf_counter() - target
                            if behind <= 1.0 / fps or skips >= max_skip:
                                break
                            frame_index += 1
                            skips += 1
                        if not ok:
                            break
                        dropped += skips
                    step_once = False
                    t_frame = frame_index / fps
                    res = p.step_frame(frame, frame_index, t_frame, fps,
                                       extra_latency_ms=render_ms)
                    p.account(res, frame_index, t_frame)
                    view = p.annotate(frame, res, extra_lines=self._hud_extra(
                        res, speed, paused, recording, show))
                    r0 = time.perf_counter()
                    if show:
                        cv2.imshow(window, view)
                    render_ms = (time.perf_counter() - r0) * 1000.0
                    if sink:
                        sink.write(view)
                    if fh:
                        fh.write(json.dumps(res["rec"]) + "\n")
                    frame_index += 1
                    if not quiet and p.n_processed - last_report >= progress_every:
                        last_report = p.n_processed
                        st = p.stats
                        print(f"  {p.n_processed:6d} frames  "
                              f"{res['reading'].state:8s} miss={res['reading'].misses:3d}  "
                              f"confirmed {st.sustained} ({st.sustained_pct:.1f}%)  "
                              f"fallback {st.fallback}  yolo-calls {st.redetect_calls}  "
                              f"loop {p.loop_ms_ema:.0f}ms", flush=True)
                    if max_frames and p.n_processed >= max_frames:
                        break

                delay = 0.0
                if not paused and not p.is_live and speed > 0:
                    target = t0 + (frame_index / fps) / speed
                    delay = (target - time.perf_counter()) * 1000.0
                if show:
                    # keep the GUI pumped and the pace accurate at the same time:
                    # waitKey has the same coarse granularity as sleep, so use
                    # short slices and keep the last key seen.
                    key = -1
                    deadline = time.perf_counter() + max(delay, 1.0) / 1000.0
                    while True:
                        k = cv2.waitKey(2) & 0xFF
                        if k != 255:
                            key = k
                        if time.perf_counter() >= deadline:
                            break
                else:
                    key = -1
                    if delay > 0.5:
                        _sleep_until(time.perf_counter() + delay / 1000.0)
                speed, paused, step_once, quit_ = self._key(key, speed, paused)
                if quit_:
                    break
                if paused and step_once:
                    continue
        except KeyboardInterrupt:
            print("\n[live] interrupted")
        finally:
            p.stats.close(frame_index)
            if sink:
                sink.close()
            if fh:
                fh.close()
            cap.release()
            if show:
                try:
                    cv2.destroyWindow(window)
                except Exception:
                    pass

        wall = time.perf_counter() - t0
        p.stats.print_report("[live] ")
        if dropped:
            print(f"[live] dropped {dropped} source frames to stay real-time "
                  f"(raise --speed, lower --imgsz, or use a smaller model)")
        stats_path = p._out_path("stats", "_live_stats", ".json")
        summary = {
            "ok": True,
            "mode": "live" if p.is_live else "playback",
            "source": str(self.source),
            "frames_processed": p.n_processed,
            "dropped_frames": dropped,
            "wall_seconds": round(wall, 2),
            "playback_fps": round(p.n_processed / wall, 2) if wall > 0 else None,
            "avg_detect_ms": round(p.det_ms_ema, 3),
            "avg_loop_ms": round(p.loop_ms_ema, 3),
            "measured_latency_ms": round(p.latency.ms, 3),
            "detect_every": p.detect_every,
            "backend": p.backend,
            "model": str(cfg["detector"]["model"]),
            "imgsz": cfg["detector"]["imgsz"],
            "tracker": cfg["tracker"]["name"] if p.use_botsort else "own-association",
            "fallback": p.stats.summary(),
            "final_state": p.tracker.state if p.tracker else None,
            "stats_path": str(stats_path),
        }
        with open(stats_path, "w") as f2:
            json.dump(summary, f2, indent=2)
        print(f"[live] stats     {stats_path}")
        return summary
