# Edge Vision Tracker

Single-subject vision tracking for an edge device, in two modes:

**live** the preview window plays the footage in real time and the box, trail and
counters update every frame. This is the mode it is built for; point it at a
camera or a stream and it runs the same way.

**batch** process a file as fast as possible and write an annotated mp4 plus a
per-frame jsonl for tuning.

It also answers one question directly: **how often does the tracker have to fall
back on YOLO?** The fallback accounting counts every frame the tracker sustained
the subject from detections, every frame it had to fall back on its own motion
model, and every time it had to ask YOLO to re-establish the track.

Built to the Tab 2 spec (`02_Software & Vision Tracking`), sections 2.1 - 2.4.

```
source ─► YOLO (per-frame, stateless) ─► BoT-SORT (IDs, GMC, Re-ID)
   file / camera / rtsp                         │
                                     primary subject lock
                                LOCKED ─► COASTING ─► LOST ─► re-ID relink
                                                │
                                6-state Kalman + 1/kHz interpolation
                                                │
                          live window + fallback counters (or mp4 + jsonl)
```

## Live mode

```bash
python run.py yourclip.mp4 --play          # real-time playback of a file
python run.py 0 --play                     # camera index 0
python run.py 'rtsp://<user>:<password>@<camera-ip>:554/stream' --play
```

Click the object you want tracked, press Enter, and it plays at the source frame
rate from then on. On screen: the subject box (green when sustained, amber while
coasting, red when lost), its trail, the velocity arrow, every other detection
faintly, and the running counters.

Keys while the window has focus:

```
space    pause / resume
s        step exactly one frame
r        reset the fallback counters
h        toggle the HUD
p        toggle trail style: line <-> tracking points
d        toggle drawing every detection, not just the subject
+ / -    playback speed (files only), 0 = as fast as possible
q, esc   quit
```

Headless (no display, e.g. on the device itself), still paced in real time:

```bash
python run.py 0 --play --no-window --stats out/cam0_stats.json
python run.py yourclip.mp4 --play --no-window --speed 0     # as fast as possible
```

Recording the preview while it plays is `--record out/session.mp4`. Writing
per-frame records is `--jsonl out/session.jsonl`.

If the machine cannot keep up, the runner drops source frames rather than lagging
behind (so the view stays live) and tells you how many it dropped at the end.
Raise `--imgsz 320`, use a smaller model, or run `--detect-every 2`.

## The fallback accounting

Per frame, for the tracked subject:

- **sustained** the subject came back from the tracker with an associated
  detection. BoT-SORT carried it forward from YOLO, nothing had to fall back.
- **coast** no box for the subject this frame, so the motion model (Kalman)
  supplied the pose. Within `tracker.coast_max_frames` the same track can resume.
- **lost** the coast budget ran out, so the pipeline asked **YOLO** again for a
  full-frame, low-confidence re-detect. This is the literal fallback.
- **reacquire** the subject reappeared after a gap.
- **id switch** it reappeared under a *different* track ID, meaning BoT-SORT did
  not sustain the track and had to create a new one.

Reported live in the window and at exit:

```
[live] YOLO-confirmed 1284 (94.3%)  fallback 78 (5.7%)
[live]   coast 62  lost 16  max gap 34f
[live]   yolo re-detect calls 4  reacquired 5  id switches 1
[live]   new track ids 3  live ids 9  det/frame 2.1
[live]   longest unbroken YOLO run 412f  window 96% confirmed
```

Written to `out/<name>_live_stats.json` (batch mode: `_track.summary.json` under
`"fallback"`), including the start/end frame of every gap so you can jump to the
exact moment the tracker struggled.

**To measure how long BoT-SORT holds without the detector**, run the detector
every Nth frame:

```bash
python run.py yourclip.mp4 --play --detect-every 3
```

Every frame is still processed and the box is still drawn, but YOLO only runs on
every third one. If sustained stays high, the tracker is genuinely carrying the
subject; if it collapses, the tracker was only ever riding on per-frame
detections. Note the caveat: with `--detect-every N`, BoT-SORT sees N times fewer
calls, so its `track_buffer` effectively covers N times longer.

## What each spec section maps to

**2.1 Detection & feature extraction** - `vt/detect.py`
- `UltralyticsDetector` (`.pt`, INT8 `.onnx`, `.tflite`, NCNN, OpenVINO) and
  `OnnxDetector` (onnxruntime only, letterbox + NMS in NumPy, decodes YOLOv5 /
  v8 / v11 / NMS-free end-to-end heads).
- Pose keypoints are carried through and drawn with `--keypoints` (use
  `--model yolov8n-pose.pt`).
- Monocular depth in `vt/geometry.py`: pinhole math, focal length from HFOV or a
  calibrated value, IPD depth (`z = f * X / x_px`), plus a pseudo-depth fallback
  from a known body height.

**2.2 YOLO + BoT-SORT** - `vt/detect.py`, `vt/gmc.py`, `vt/tracker.py`
- YOLO is stateless per frame; BoT-SORT (`configs/botsort_reid.yaml`) carries
  track IDs and trajectories. Re-ID is switched on (ultralytics ships it off) and
  `track_buffer` is raised to 60.
- GMC: BoT-SORT's own sparse-optical-flow GMC runs when BoT-SORT is active, and
  the pipeline disables its duplicate so a panning camera is not compensated
  twice. On the ONNX/TFLite path (no BoT-SORT), `vt/gmc.py` provides the same
  thing: Shi-Tomasi corners, LK flow, RANSAC partial-affine, with sanity limits
  on shift and scale so a scene cut cannot warp the track into the void.
- IoU association + appearance distance: candidates are scored on
  `w_iou*(1-IoU) + w_emb*(1-sim) + w_motion*mahalanobis`, with a bonus for a
  matching track ID.
- Spatial reliability mapping: the descriptor applies an elliptical weight mask
  over the box so background corners do not contaminate the template.
- Scale / aspect adaptation: box size is EMA-smoothed and the search window grows
  with the number of missed frames while coasting.
- Optical flow: `--flow-refine` re-seeds LK points inside the subject box every
  frame and nudges the box between detector frames.
- DCF/CSRT/KCF are deliberately not used: they live in opencv-contrib and would
  be a second tracker fighting the Kalman filter. LK + Kalman covers the need
  with no extra dependency.

**2.3 State estimation & latency decoupling** - `vt/kalman.py`, `vt/sync.py`
- 6-state Kalman `[x, y, z, vx, vy, vz]^T`, discrete white-acceleration process
  model so `Q` scales correctly with `dt`, Mahalanobis gating for association,
  uncertainty inflation while coasting.
- `TrajectoryBuffer` interpolates between vision samples and extrapolates on the
  last velocity only within `max_extrapolate_ms` (default 120 ms); past that it
  holds rather than flying off.
- `LatencyModel` tracks end-to-end latency (perception plus presentation in live
  mode). Every record carries `target_now`, the pose predicted for the moment the
  actuator acts.
- `tools/motor_stream.py` replays a record at 1 kHz to prove the path:
  `python tools/motor_stream.py --jsonl out/yourclip_track.jsonl --hz 1000`.

**2.4 Dynamic reference updating & online Re-ID** - `vt/tracker.py`, `vt/reid.py`
- Gallery + anchor at selection time: a fixed anchor embedding, a capped gallery,
  and an EMA copy. Matching takes the max similarity over all of them, which is
  what makes a target recognisable again after it changes pose.
- Update strategies: EMA blending with a small alpha and score-based gating
  (`update_score_gate`, 0.50 default; `--profile strict` sets the spec's 0.95).
- Template drift / avalanche guard: updates freeze when the candidate is too
  dissimilar to the EMA, or when it has diverged from the **anchor** past
  `drift_floor`. Rotation or a crossing cannot poison the template; the lock
  coasts instead of absorbing a corrupted view.
- Dual-class (person + face): `vt/face.py` runs a frontal-face probe inside the
  person box (OpenCV's bundled Haar cascade, no extra model), giving an
  independent cue and the inter-pupillary pixels for metric depth.
- Coasting 1-30 frames: Kalman constant-velocity prediction, growing search
  window, optional LK refinement.
- Re-detection recovery >30 frames: `LOST` triggers the full-frame low-confidence
  re-detect (counted in the fallback accounting), then re-links by gallery/EMA
  embedding plus motion prior, keeping the original identity.

## Install

### Windows

`.venv` is **not** in the zip on purpose. A virtual environment is not portable:
it hard-codes the interpreter path and contains native binaries for the OS it was
built on, so a macOS `.venv` is unusable on Windows. Build a fresh one:

```powershell
# PowerShell, from inside the vision_tracker folder
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

```
REM or cmd
setup_windows.bat
```

Either script creates `.venv`, installs the requirements and the ultralytics
backend, then prints a version check. Then:

```
.venv\Scripts\python.exe run.py yourclip.mp4 --play
```

Windows puts the interpreter in `.venv\Scripts\python.exe`, not
`.venv/bin/python` as on macOS and Linux; every `.venv/bin/python` in this README
becomes that. Activate once with `.venv\Scripts\activate.bat` to use plain
`python`.

Two Windows notes. The `numpy<2` pin in `requirements.txt` exists only for Intel
macOS; on Windows x64 it is harmless, leave it. And `ffmpeg` is optional: without
it the pipeline falls back to the OpenCV mp4v writer, which plays locally but is
less portable than h264, so `winget install Gyan.FFmpeg` if you want shareable
output.

For the edge device, prefer the ONNX INT8 path on x86-64:

```
.venv\Scripts\python.exe tools\export_int8.py --model yolov8n.pt --to onnx-int8 ^
    --calib-video yourclip.mp4 --imgsz 320 --out models\
.venv\Scripts\python.exe run.py yourclip.mp4 --play --backend onnx --model models\yolov8n_int8.onnx --imgsz 320
```

### macOS / Linux

```bash
cd ~/Projects/vision_tracker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv, pyyaml, onnxruntime
pip install "ultralytics>=8.2,<9"        # optional but recommended: BoT-SORT + .pt models
```

Go straight to live mode, it needs no detections at all:

```bash
.venv/bin/python tests/smoke_test.py
```

## Selection, when there is no mouse

```bash
python run.py yourclip.mp4 --play --no-gui --select auto           # biggest detection
python run.py yourclip.mp4 --play --select point --point 640,360   # by pixel
python run.py yourclip.mp4 --play --select roi --roi 500,200,180,360
python run.py yourclip.mp4 --play --select-at 12.0                 # click at 12s, not 0s
```

Every run saves the selection frame as `out/<name>_selection.png` with the
candidate boxes and their indices drawn, so you can pick a `--point` coordinate
off that image when there is no display.

`python run.py --print-config` dumps the fully merged config (defaults + yaml +
profile + CLI flags).

## Edge deployment (INT8)

The detector is the only expensive part. Preference order for an edge box: TFLite
INT8 (Pi/Android/NPU stacks), ONNX INT8 statically calibrated on your own clip,
NCNN/OpenVINO if the vendor wants them, then ONNX FP32.

```bash
python tools/export_int8.py --model yolov8n.pt --to onnx-int8 \
    --calib-video yourclip.mp4 --imgsz 320 --out models/
python tools/export_int8.py --model yolov8n.pt --to tflite-int8 --imgsz 320
```

Calibrate on your deployment footage, not COCO, so the activation ranges match.
`yolov8n` at 320 is the sweet spot for 30 fps on ARM-class CPUs; `yolov9t` or
`yolov10n` are drop-in if you want more accuracy. Keep `imgsz` a multiple of 32.
The Re-ID embedder defaults to the built-in `lite` descriptor so you are not
paying for a second CNN per frame; `--embedder mobilenet --reid-weights
mobilenetv3.onnx` swaps in 576-d CNN features if there is headroom.

Frame budget reference, measured on a 1.4 GHz Intel i5 with no GPU:
`yolov8n` at 416 = about 65 ms per frame (12 fps); the `lite` embedder, Kalman
and association together are under 11 ms; rendering the HUD and trail is about
8 ms. So on that box the detector is ~80% of the frame cost and INT8 plus a
smaller `imgsz` is where the win is.

## Tuning for your clip

Everything is in `config.yaml`; CLI flags override it.

| symptom | knob |
|---|---|
| loses the subject during fast motion | `tracker.iou_min` down, `tracker.search_growth` up, `--flow-refine` |
| grabs a bystander / lookalike | `tracker.sim_min` up, `reid.update_score_gate` up, `--profile strict` |
| re-ID fails after occlusion | `tracker.sim_min_lock` down, `reid.gallery_size` up, `reid.drift_floor` down |
| template drifts onto the background | `reid.update_score_gate` up, `reid.drift_floor` up, `reid.ema_alpha` down |
| should coast longer before giving up | `tracker.coast_max_frames` up |
| too slow on the device | `detector.imgsz` down, INT8 export, `--detect-every 2`, `face.every_n: 2` |

In live mode the counters are the instrument: healthy is mostly sustained with
short coast gaps and no id switches. The jsonl from batch mode is the second
instrument, chart `sim` and `misses` against time to see exactly where the lock
struggles.

## Outputs

`*_track.jsonl`, one record per frame: `state`, `bbox`, `center`, `velocity`,
`depth_m`, `score`, `sim`, `iou`, `maha`, `misses`, `measured`, `relinked`,
`detected` (did the detector run this frame), `det_ms`, `loop_ms`, `latency_ms`,
`target_now` (latency-compensated position), and `detections` (every detection
that frame with its track ID).

`*_live_stats.json` / the `fallback` block of `*_track.summary.json`: the
sustained/fallback accounting described above, with every gap's frame range.

## Tests

```bash
.venv/bin/python tests/smoke_test.py        # coast/LOST/re-ID, no model needed
.venv/bin/python tests/live_test.py         # live mode, real-time pacing, accounting
.venv/bin/python tests/onnx_test.py         # ONNX decode, letterbox, NMS
.venv/bin/python tests/selection_test.py    # --select modes + the real CLI
.venv/bin/python tests/real_yolo_test.py    # real YOLO + BoT-SORT, batch and live
```

On Windows use `.venv\Scripts\python.exe tests\smoke_test.py` and so on. Run
`smoke_test.py` first on a new machine: no model download, no GPU, no window, so
it separates "the install is broken" from "the tracking behaves oddly".

`smoke_test.py` and `live_test.py` generate a clip where the subject is fully
hidden behind a pillar for 36 frames (longer than the 30-frame coast budget)
while a lookalike stays visible the whole time, then assert that the lock coasted,
fell to `LOST`, re-linked on reappearance, never accepted the lookalike, and that
the fallback counters match what happened frame by frame. `real_yolo_test.py`
does the same on a real photo with real YOLO detections and a panning camera, in
both batch and live mode.

## Files

```
run.py                    CLI (--play for live, default is batch)
config.yaml               every tunable
configs/botsort_reid.yaml BoT-SORT with Re-ID enabled
setup_windows.bat/.ps1    one-shot Windows environment setup
vt/live.py                live runner: real-time preview, keys, pacing
vt/stats.py               fallback accounting (sustained / coast / lost / re-ID)
vt/pipeline.py            per-frame pipeline, sources, batch runner
vt/detect.py              YOLO backends (ultralytics / onnxruntime / dummy)
vt/tracker.py             primary subject lock, gallery, drift guard
vt/kalman.py              6-state constant-velocity filter
vt/reid.py                appearance descriptors (lite / mobilenet)
vt/gmc.py                 global motion compensation
vt/flow.py                Lucas-Kanade refinement
vt/face.py                person+face probe, IPD source
vt/geometry.py            boxes, pinhole/IPD depth, masks, search windows
vt/sync.py                trajectory buffer, latency model, motor-rate sampling
vt/render.py              overlays, HUD, fallback readout
vt/select.py              click / point / roi / auto selection
tools/export_int8.py      ONNX/TFLite INT8 export + calibrated quantisation
tools/motor_stream.py     replay a record at motor rate
tools/make_test_video.py  synthetic clip + ground truth for the tests
```
