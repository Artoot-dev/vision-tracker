# PROJECT LEDGER: vision tracker

Repository: https://github.com/Artoot-dev/vision-tracker (public, branch main)

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

---

## 7. Session additions, ingest of Summary1 and Summary2 (another AI conversation)

### 7.1 Verified: the Stride-32 rule is real, and stricter than claimed

Measured on yolo26n fp32 ONNX, 1080p source, this CPU:

| tensor | divisibility | result |
| --- | --- | --- |
| 640x480 | both /32 | 28.43 ms, 92.6 us per kpx |
| 640x640 | both /32 | 38.01 ms, 92.8 us per kpx |
| 768x576 | both /32 | 40.70 ms, 92.0 us per kpx |
| 960x544 | both /32 | 48.37 ms, 92.6 us per kpx |
| 640x448 | both /32 | 26.45 ms, 92.2 us per kpx |
| 720x540 | neither | export silently became 736x544, then runtime INVALID_ARGUMENT |
| 960x540 | height not /32 | same: export silently reshaped, runtime refused |

Conclusions:

1. Cost is linear in tensor area to within 0.9 percent, 92.0 to 92.8 us per kilopixel
   across every valid shape. So the Summary2 statement that cost is O(N squared) is
   correct, and the constant is 92 us per kpx on this machine.
2. A non multiple of 32 is not merely slower, it is silently altered at export and then
   rejected at runtime. Ultralytics rounds the request up (720x540 became 736x544) and the
   resulting static graph no longer matches the requested input.
3. Rule adopted: every tensor height and width must be divisible by 32, and the exporter
   output shape must be asserted against the request, never assumed.

### 7.2 Corrected: the widescreen letterbox argument is backwards

`letterbox` preserves aspect ratio, so a 1920x1080 (16:9) frame uses this share of each tensor:

- 640x640, aspect 1.0: content 640x360, 56.2 percent used, 43.8 percent padding
- 640x480, aspect 1.33: content 480x270, 42.2 percent used, 57.8 percent padding
- 768x576, aspect 1.33: content 576x324, 42.2 percent used, 57.8 percent padding
- 960x544, aspect 1.76: content 544x306, 31.9 percent used, 68.1 percent padding

So on a 16:9 feed the square 640x640 wastes 43.8 percent and the widescreen 640x480 wastes
57.8 percent. The square tensor is closer to 16:9 than 4:3 or 1.76 is, so Summary2 has the
direction of the trade inverted. To actually exploit a 16:9 feed the tensor must be 16:9
itself, for example 608x352 or 640x352, both divisible by 32.

Also, a widescreen tensor at equal long side is not cheaper per useful pixel. 640x480 holds
270 content rows at 92.3 us per kpx, while 640x640 holds 360 content rows for 38.01 ms, so
each content row costs 105 us against 77 us, so the square tensor gives more useful rows
per unit time despite the padding.

Not yet measured: a 16:9 tensor (for example 608x352 or 640x352, both divisible by 32) or a
direct crop, which fills the tensor with no padding at all. Both are scheduled as tests, not
assumptions.

### 7.3 Adopted from the other conversation

1. Coordinate normalisation before the tracker, so Kalman velocity and IoU are immune to crop
   and zoom scaling. This matches decision 10 in section 1, independently arrived at.
2. Reject raw pixel coordinates from dynamic crops, for the same reason.
3. Reject runtime dynamic tensor shapes for TensorRT class engines: reallocation and kernel
   re-optimisation cause latency spikes. This is the same argument as decision 13, fixed
   tensor phase ladder.
4. Reject sub-4-bit or log profile input; keep 8-bit sRGB per channel, which is 24-bit total
   and is what detectors are trained on. Terminology resolved: 8-bit means per channel.
5. Keep 1:1 as the preferred mapping, and now also allow explicit upscaling as a separate
   opt-in flag (see section 7.4), rather than a silent clamp.
6. Stride-32 rule, hardened per 7.1.
7. Dual static model discovery and tracking, which is our phase ladder under another name.
8. Domain specific INT8 calibration on 300 or more frames, already open question 2.

### 7.4 Conflicts, resolved by measurement, in favour of this ledger

1. Summary2 rejects upsampling a small ROI into a larger tensor. Keep it, but only as an
   explicit, measured, opt-in tool. Rationale: a 24 px person was detectable at 1:1 in a small
   crop (IoU 0.83) and undetectable in the full frame. Upscaling invents no detail, so it is
   never a substitute for a real crop; it only moves a subject back into the detector's trained
   apparent size band. It is now `allow_upscale`, defaulting off.
2. Summary2 rejects confidence normalisation for INT8. Agreed as a matter of principle: fix
   calibration first. But thresholds must still be retuned per precision, because measured
   confidence drops 0.2 to 0.3 even with correct calibration.
3. Summary1 treats compression as a live component. It is out of scope for the perception path,
   decision 17. Only relevant to a networked source or the exported file.
4. Summary1 says disable Re-ID and GMC when stationary or compute bound. GMC must stay: it is
   the reason identity survives a moving gimbal. Re-ID is already rejected.
5. Any nvvideoconvert, DALI, CUDA, or TensorRT specific advice assumes an NVIDIA GPU. The edge
   device is still undecided, so those benchmarks cannot be run and are not scheduled.

### 7.5 Test results on the Summary2 pseudocode

The Stride-32 and letterbox claims are now measured (7.1, 7.2). The 1:1 floor claim is
contradicted by the 24 px detection at a 128 crop (section 2), so the floor must be expressed
as subject pixels in the tensor, not as a crop minimum.


---

## 8. Correction: my section 7.2 was wrong, and the log was right after all

Two errors in my own previous entry. Recording them because the human asked for
mistakes to be flagged, not quietly overwritten.

### 8.1 What I got wrong

1. Orientation labels. I passed `imgsz=(h, w)` and then wrote the results out as if they
   were `(w, h)`. So the shapes I called "widescreen" were in fact 576x320 and 640x352, which
   are PORTRAIT, aspect 0.56 and 0.55. I never tested a true 16:9 tensor at all.
2. The conclusion. I wrote that a square 640x640 is "closer to 16:9" and therefore better than
   a widescreen tensor. That is arithmetically false. The share of a tensor used by a width
   bound letterbox is 0.5625 x (tensor width / tensor height), so utilisation is linear in the
   tensor's own aspect, and 16:9 fills it completely. Square is not closer to anything; it
   throws away 43.8 percent of its area.

Consequence: the Summary2 claim that a widescreen tensor wastes less than a square one was
correct in direction all along. Its specific tensor choice, 640x480, was still wrong, because
640x480 is 4:3 (aspect 1.33), not 16:9.

### 8.2 What is actually true, measured

For a 1920x1080 feed, all shapes divisible by 32, no crop:

| tensor | aspect | tensor kpx | useful kpx | used pc | latency | us per useful px |
| --- | --- | --- | --- | --- | --- | --- |
| 640x640 | 1.000 | 409.6 | 230.4 | 56.2 | 36.20 ms | 0.1571 |
| 512x288 | 1.778 | 147.5 | 147.5 | 100.0 | 16.08 ms | 0.1090 |
| 1024x576 | 1.778 | 589.8 | 589.8 | 100.0 | 58.47 ms | 0.0991 |
| 640x352 | 1.818 | 225.3 | 220.4 | 97.8 | 21.97 ms | 0.0997 |
| 640x384 | 1.667 | 245.8 | 230.4 | 93.8 | 24.08 ms | 0.1045 |

1. Matching the source aspect fills the tensor completely. 512x288 uses 100 percent of its
   area against 56.2 percent for 640x640.
2. Per useful pixel, 512x288 is 1.44x cheaper than 640x640 (0.1090 against 0.1571).
3. At an equal latency budget of about 16 ms, a 16:9 tensor carries 147 kpx of scene while the
   largest affordable square, 416x416, carries 97 kpx. So at equal cost, matching the aspect
   covers 1.5x more of the scene.
4. A near 16:9 shape, 640x352 at 97.8 percent used, is essentially tied with exact 16:9 and is
   the better choice when more detail is wanted at 21.97 ms.
5. Recall at equal in-tensor subject height (48 px), three positions: exact 16:9 hit 3 of 3,
   and every other shape hit 2 of 3. Suggestive, not conclusive, n=1 per position.

### 8.3 What this changes in the design

1. Aspect must be chosen per state, because the crop shape differs per state. SEARCH feeds the
   whole frame, so its tensor should be 16:9: use 512x288. LOCKED crops a square ROI, so a square
   tensor is already zero padding: keep 256 or 384 square. A single shape for both is wrong.
2. This is what the "dual static model" split was groping toward in the other log. It is right,
   but it has to be dual in ASPECT, not merely in size.
3. 640x640 should not be the default anything. It is the worst of the shapes measured for a
   widescreen feed.
4. Still untested: a direct crop that fills the tensor with no padding, and the exact small-subject
   floor per aspect. Both are now scheduled tests rather than assumptions.

## 9. Provenance: every external claim from Summary1 and Summary2 is assumption

The human confirmed that the conversations behind those logs never ran a test. That reframes
them, and one detail proves the point: Summary2 quotes INT8 latency as 9.2 ms rising to 18.5 ms
on a non VNNI CPU. Those are byte identical to numbers measured in THIS project's session and
fed back out. So Summary2 is partly a restatement of our own results, not independent evidence.
Echoed numbers are the specific hazard of splitting one project across several AI conversations.

Tally of every substantive claim in the two logs, against this machine:

VERIFIED by our own measurement, and kept:
1. Inference cost is driven by input tensor area, not ROI content. O(N squared) with a constant
   of 92.0 to 92.8 us per kilopixel.
2. Stride-32 divisibility is mandatory, and stricter than claimed: a violation is silently
   reshaped at export and then rejected at runtime.
3. INT8 on a non VNNI CPU doubles latency. But see above: this is our own number, not a
   corroboration.
4. Coordinate normalisation before the tracker, and rejection of raw crop coordinates.
5. Rejection of runtime dynamic tensor shapes.
6. 8-bit input means 8 bits per channel, 24 bits total. True by definition.
7. Matching the tensor aspect to the source reduces waste. Right in direction, wrong tensor
   given, see section 8.

CONTRADICTED by our measurement, so dropped:
1. A square tensor "wastes up to 25 percent". It wastes 43.8 percent.
2. Upsampling a small ROI into a larger tensor is never useful. Contradicted: a 24 px subject was
   detected at 1:1 inside a small crop and was undetectable in the full frame, so crop plus
   explicit scaling is a real tool.
3. Cost per content row favours a widescreen tensor. Backwards; per useful pixel the aspect
   matched tensor wins by 1.44x.
4. Disable GMC when compute bound. GMC is what makes identity survive a moving gimbal, and it
   costs 2.11 ms full frame.
5. "Log Entry" statuses such as Verified and Validated Strategy in the human's own vault. None of
   them correspond to a run. They should be downgraded to Idea until a test exists.

UNFALSIFIABLE HERE, parked, not to be built on:
1. TensorRT reallocation stalls, DALI, CV-CUDA, nvvideoconvert, VRAM local cropping: all
   NVIDIA GPU specific, and the edge device is undecided.
2. Bitrate and compression policy: out of scope on the perception path.
3. INT8 confidence drop resolving to under 0.05 with 300 or more calibration frames. Plausible,
   untested either way.

Rule adopted from this: an external log enters as UNVERIFIED by default, and its status chips
carry no weight. A claim is promoted to verified only when a test on this machine reproduces it,
and the ledger records which test.

