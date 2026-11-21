# PROJECT LEDGER: vision tracker

Purpose: the single durable source of truth for decisions and measured numbers, so any AI
conversation (this one, or another) can be brought up to speed from this file instead of
from chat history. Chat history gets compacted and silently loses numbers. This file does not.

Requirements live in the human's own spec (Tab 1 to Tab 5). This file records decisions,
measurements, and open questions only.

---

## 0. How to ingest a conversation log from another AI

The human pastes the log raw, unedited. No cleanup needed. In return, this file gets updated
and the reply uses this fixed five part shape, so the human can carry the last part back to
the other conversation:

1. ADOPT. Decisions in the log that are now treated as settled. Quoted in one line each.
2. CONFLICT. Anywhere the log disagrees with a measured number below. The number wins, and
   the direction this project will take gets stated.
3. VERIFY. Claims that are plausible but not yet tested, each paired with the exact test that
   would settle it.
4. MISSING. Parameters needed before implementation, with units and expected format.
5. ASK THE OTHER CHAT. Three to seven questions, each answerable with a number or a yes or no,
   never an essay prompt.

Rule: never re-litigate an item in section 6 (superseded) without new measurements.

---

## 1. Settled decisions

1. Target hardware class is an edge device, not a desktop. All budgets are per frame at 30 fps,
   so 33.3 ms total.
2. Model: switch from yolov8n to yolo26n. Conditional on one recheck, see section 4.
3. Precision: FP32 ONNX is the portable default. Precision is a config value, not a hardcode.
4. Detector cadence: run detection on a throttle, default every 5th frame (167 ms), with the
   tracker carrying the frames between. On loss, raise the cadence to about 100 ms.
5. Webcam identity is ours and never renumbers. BoT-SORT IDs are internal bookkeeping only.
6. Correctness beats cost. Tracking the right subject outranks minimising compute.
7. Identity is decided by a frozen anchor plus an enrolled gallery. A running EMA may adapt for
   smoothness but may never decide identity.
8. Re-link uses a cascade, cheapest test first: class and confidence, then spatial radius,
   then size and aspect, then the appearance descriptor. The descriptor runs on survivors only.
9. Re-link acceptance is relative: the best candidate must beat the runner-up by a margin,
   not just pass a fixed threshold. This is what structurally prevents a nearest-box swap.
10. All internal state lives in normalised global frame coordinates (0 to 1). Conversion to
    and from ROI pixels happens only at the detector boundary.
11. Always crop from the original frame, never from a previous crop.
12. ROI geometry is owned by the tracker state, not by the frame, and it expands while searching.
    A detection miss inside a ROI is not evidence of global absence, so the LOST state broadens
    to full frame.
13. Phase ladder of fixed tensors: LOCKED uses a 1:1 ROI at 256, 384 or 512 depending on subject
    size. SEARCH uses the full frame at 320. EXHAUSTIVE uses 640 or a rolling sweep. CONFIRM uses
    a small ROI at 1:1 before any accepted re-link.
14. ROI ladder steps are 256, 384, 512, 640, with 768 as an exceptional close-up rung.
15. ROI size is roughly 2 to 4 times the subject's longest side. Sub-256 crops are allowed when
    upscaled into the tensor.
16. Tiled or sliced inference is banned on the live path. It is allowed only inside the search
    state, and only amortised across ticks.
17. Compression is out of scope for the perception path. No variable bitrate, no region coding.
18. Global motion compensation needs full frame features even when detection is ROI local.
19. The gimbal needs angular error, not distance, for pointing. Metric depth is not required for
    pan and tilt. The third state component is better spent on apparent scale, which drives zoom
    and constant-size framing.
20. A benchmark harness is a first class deliverable, and its outputs are identity retention,
    wrong-subject frames, time to correct re-link after occlusion, and YOLO fallback counts.

## 2. Measured numbers

Machine: Intel i5-8257U, 8 GB, no GPU, macOS. Model: yolo26n, FP32 ONNX, static shape, unless
stated otherwise. Treat these as the reference for an ordinary CPU edge class, which is within
about 1.5x of a Raspberry Pi 5.

Detector cost per tick, and amortised over a 5 frame cadence:

- imgsz 256: 7.54 ms, 1.51 ms amortised
- imgsz 320: 10.57 ms, 2.11 ms amortised
- imgsz 384: 15.78 ms, 3.16 ms amortised
- imgsz 512: 24.95 ms, 4.99 ms amortised
- imgsz 640: 37.44 ms, 7.49 ms amortised
- imgsz 768: 55.10 ms, 11.02 ms amortised

Cost is set by the tensor, not by the frame content:

- whole 640x480 frame letterboxed to 320: 10.64 ms
- 120x120 ROI letterboxed to 320: 10.71 ms
- 320x240 ROI letterboxed to 320: 10.67 ms
- spread 0.6 percent, so cropping does not make detection cheaper. It is a resolution and
  distractor tool, not a speed tool.

Detection floor on a full 1920x1080 frame:

- imgsz 160: nothing detected, even a 192 px tall person. Do not use 160 for full frame search.
- imgsz 320: erratic in the 8 to 32 px in-tensor band.
- imgsz 640: reliable from about 64 px in source, which is 21 px in the tensor.
- Cropping wins: a 24 px person was detected at IoU 0.83 to 0.84 when fed 1:1 inside a 256
  tensor from a 128 or 256 crop, which the full frame could not do at any size.
- Caveat: this used one pasted person crop at eight scales, and results were non monotone at the
  margins (24 px hit, 32 px missed, 48 px hit). Direction is solid, exact floor is not. The
  benchmark harness should measure the real floor on real footage.
- Governing formula: subject pixels in tensor = subject pixels in source x imgsz / crop pixels.

Tiling arithmetic, 1080p:

- 45 tiles of 256 px at 20 percent overlap, 7.66 ms each, 345 ms per sweep.
- One whole frame at 320 is 11.8 ms, at 640 is 38.0 ms.
- A sweep is 29x a 320 pass and 11x a 640 pass. Tiling cannot be faster than one pass.
- The human's 5 capture scheme (4 quadrants of half width and height plus a centre capture):
  37.7 ms at 256, 52.9 ms at 320. It sees the subject at the same size as one 640 pass
  (38.0 ms) while costing 1.4x more.
- Why: cost grows with the square of the tensor size, resolution only grows linearly, so one
  larger tensor beats N tiles at equal resolution by about the fourth root of N.

Other measured costs:

- LK optical flow, 640x480 frame: 2.11 ms with 81 features. 320x240 ROI: 0.59 ms. 160x160: 0.33 ms.
- Resize 1080p to 640x360: 0.68 ms. 1080p to 960x540: 0.73 ms.
- Resize 4K to 1080p: 3.85 ms. 4K to 640x360: 0.77 ms.
- Frame copy: 1080p 1.42 ms, 4K 5.70 ms.
- Letterbox 640x480 to 320: 0.15 ms. One ROI tile paste into a 320 composite: 0.04 ms.
- So a foveal or cropped input costs about 0.3 ms per tick, roughly 2 percent of the detector cost.

Model comparison, ONNX fp32 at imgsz 320, median of 11:

- yolov8n: 10.8 ms, 12.7 MB fp32, 3.4 MB int8
- yolo26n: 9.2 ms, 9.8 MB fp32, 2.8 MB int8. About 15 percent faster and about 60 percent of
  the size.
- In torch, the reverse: yolov8n 28.5 ms against yolo26n 32.0 ms at 320. Irrelevant for
  deployment, relevant only if training on the Mac.
- INT8 on this CPU is slower, not faster: yolov8n 10.8 to 18.7 ms, yolo26n 9.2 to 18.5 ms.
  Cause: this 2019 i5 has AVX2 but no VNNI, so quantize and dequantize nodes are pure overhead.
  INT8 only pays on hardware with an int8 datapath.
- INT8 deflates confidence: yolov8n mean person confidence 0.762 to 0.451 and one spurious extra
  detection. yolo26n 0.681 to 0.506 with the same detection count. Thresholds are therefore
  precision specific and must be retuned or normalised per precision.

Environment facts:

- Project venv pins numpy below 2. Installing ultralytics extras (onnxslim) will pull numpy 2.x
  and break torch 2.2.2 on Intel macOS. Re-pin with: pip install "numpy>=1.24,<2".
- Ultralytics replaced the old int8 and half export flags with quantize. Static INT8 ONNX:
  export(format="onnx", quantize=8, data=<yaml>). Test suites: 5 of 5 pass after the re-pin.
- pip installs trigger a package threat intelligence approval prompt that can time out and
  block the command.

## 3. Design conclusions worth keeping

- 4K does not help the detector for the same framing, because the resize to the tensor throws the
  extra pixels away. Subject pixels in tensor depend only on the subject's share of the crop.
- At 4K a box must be twice the pixel size to cover the same field of view. A 320 box at 4K is
  the same framing as a 160 box at 1080p.
- 4K pays off in exactly one way: as a zoom source, cropping tighter from the native frame so the
  subject fills more of the tensor. Otherwise downscale to a small proxy immediately.
- The real 4K bottleneck is the global per pixel work, not YOLO. A full frame LK or GMC pass at 4K
  touches 27x the pixels of the 640x480 case measured at 2.11 ms.
- Recommended 4K architecture: one downscaled proxy for all global work, native ROI crop for the
  detector. Overhead under a millisecond.
- A handheld does not need distance to point the gimbal. Angle error is enough. This is why zoom,
  not multi scale search, is how consumer gimbals extend tracking range.

## 4. Open questions

1. Edge device family. Three classes, and the only thing that changes is the export format and
   the threshold values. int8 only accelerator: INT8 mandatory, thresholds retuned. Float capable
   GPU: FP16 preferred, INT8 pointless. Ordinary CPU: stay FP32, because INT8 measured slower here.
2. INT8 accuracy recheck before the model switch is locked: calibrate on 300 or more frames of the
   human's own footage, not the 4 image coco8 sample ultralytics warns about.
3. Search window length after loss, 15 or 30 seconds. Default set to 15, configurable.
4. Primary capture resolution, 1080p or 4K.
5. Target class. The spec mentions sub 10 px targets against sky, which does not sound like a
   person, so the class list and the detector training may need to change.

## 5. Blocked

- Two code edits awaiting an approval prompt: non-square (height, width) input sizes so a wide
  tile can use a wide tensor, and an allow_upscale flag so a small ROI can be enlarged into the
  tensor. Plus the confirmation test of the 5 capture scheme against a single pass.
- GUI click to select a subject is unverified, no display control available. The headless
  equivalent is verified: --select point --point x,y.
- Windows setup scripts are reviewed but never executed.
- A real mouse driven session cannot be exercised here.

## 6. Superseded, do not revisit without new measurements

- BoT-SORT Re-ID. Rejected by the human on compute and reliability grounds. Note the ultralytics
  configuration with_reid plus model auto resolves to a pass through that reuses detector features,
  so it costs no second CNN. Rejected anyway.
- Coasting for 1 to 30 frames, and 3 frame velocity estimation. Superseded by: no coasting past
  the last estimate, and a Kalman filter that pools velocity evidence across many ticks.
- LOG-003's reasoning that an ROI crop removes 80 to 90 percent of frame data and therefore saves
  time. The conclusion about a downscaled proxy is right, the reasoning is wrong (0.6 percent).
- imgsz 160 for full frame search at 1080p. Measured to detect nothing.
- Tiled inference as a speed win. It is an accuracy technique with a linear cost multiplier.
- An assistant claim that a 320 ROI at 4K matches a 640 ROI at 1080p. Wrong direction, see
  section 3.
- An assistant claim that a 128x128 ROI at imgsz 256 counts as a 2x upscale. Wrong: the letterbox
  clamps to never enlarge unless allow_upscale is set, so it was fed 1:1 inside a padded tensor.
- Reading "balanced" in config.yaml as anything to do with compression. It is an association gate
  strictness profile only. The strict profile raises the template update gate from 0.50 to 0.95.
