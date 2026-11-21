"""Detection backends.

auto        pick ultralytics if installed, else onnx, else fail loudly
ultralytics YOLO(.pt/.onnx/_int8.onnx/_int8.tflite/_ncnn/OpenVINO) + model.track()
            -> this is the path that gives you BoT-SORT track IDs directly
onnx        onnxruntime only, letterbox + NMS in NumPy. No torch on the device.
dummy       ground-truth reader used by the smoke test (no model at all)

All backends return the same Detection list, so the tracker does not care which
one is live. INT8 quantisation is handled at export time (see tools/export_int8.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    xyxy: np.ndarray
    score: float = 1.0
    cls_id: int = 0
    track_id: int | None = None
    cls_name: str = ""
    extra: dict = field(default_factory=dict)

    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), stride=32, auto=False,
              allow_upscale=False):
    """Letterbox to exactly `new_shape`.

    auto=False is the default and the correct setting for fixed-shape exported
    models (ONNX/TFLite): the result is exactly new_shape x new_shape. auto=True
    reproduces ultralytics' minimum-rectangle padding, which rounds the padded
    size down to a stride multiple and will not match a [1,3,S,S] graph input.

    allow_upscale=False (default) never enlarges, so a small crop sits at native
    scale inside a padded tensor. allow_upscale=True resizes the crop up to fill
    the tensor, putting a small subject back into the apparent size band the
    detector was trained on. It adds no real detail.
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not auto and not allow_upscale:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2.0, dh / 2.0
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def nms_np(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


class BaseDetector:
    kind = "base"

    def detect(self, frame, frame_index=0):
        raise NotImplementedError

    def detect_track(self, frame, frame_index=0):
        """Same as detect(); backends with a real MOT overlay attach track_id."""
        return self.detect(frame, frame_index)

    def close(self):
        pass


class UltralyticsDetector(BaseDetector):
    """YOLO via ultralytics. Handles .pt, INT8 .onnx, INT8 .tflite, NCNN, OpenVINO."""

    kind = "ultralytics"

    def __init__(self, model="yolov8n.pt", imgsz=416, conf=0.25, iou=0.5, classes=(0,),
                 device="", half=False, max_det=50, tracker_cfg=None, verbose=False,
                 track=False):
        from ultralytics import YOLO  # imported lazily: torch is optional

        self.model = YOLO(model)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = list(classes) if classes else None
        self.device = device or None
        self.half = bool(half)
        self.max_det = int(max_det)
        self.tracker_cfg = tracker_cfg
        self.verbose = bool(verbose)
        self.use_track = bool(track)
        self._persist = False
        names = getattr(self.model, "names", None) or {}
        self.names = names

    def _kw(self):
        kw = dict(imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                  max_det=self.max_det, verbose=self.verbose)
        if self.classes is not None:
            kw["classes"] = self.classes
        if self.device:
            kw["device"] = self.device
        if self.half:
            kw["half"] = True
        return kw

    def _run(self, frame, frame_index=0):
        if self.use_track:
            if self.tracker_cfg:
                res = self.model.track(frame, persist=self._persist, tracker=self.tracker_cfg,
                                       **self._kw())
            else:
                res = self.model.track(frame, persist=self._persist, **self._kw())
            self._persist = True
        else:
            res = self.model.predict(frame, **self._kw())
        return res[0]

    def _to_dets(self, r):
        out = []
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return out
        b = r.boxes
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        ids = b.id.cpu().numpy().astype(int) if getattr(b, "id", None) is not None else None
        kps = None
        if getattr(r, "keypoints", None) is not None and r.keypoints is not None:
            try:
                kps = r.keypoints.xy.cpu().numpy()
            except Exception:
                kps = None
        for i in range(len(xyxy)):
            tid = int(ids[i]) if ids is not None else None
            name = self.names.get(int(cls[i]), str(int(cls[i]))) if isinstance(self.names, dict) \
                else str(int(cls[i]))
            extra = {}
            if kps is not None and i < len(kps):
                extra["keypoints"] = kps[i]
            out.append(Detection(xyxy=np.asarray(xyxy[i], dtype=np.float64),
                                 score=float(conf[i]), cls_id=int(cls[i]), track_id=tid,
                                 cls_name=name, extra=extra))
        return out

    def detect(self, frame, frame_index=0):
        return self._to_dets(self._run(frame, frame_index))

    def detect_track(self, frame, frame_index=0):
        return self.detect(frame, frame_index)


class OnnxDetector(BaseDetector):
    """onnxruntime-only YOLO. INT8 ONNX runs here without torch.

    Supports YOLOv8/v11 (1, 4+nc, N) and (1, N, 4+nc), YOLOv5 (1, N, 5+nc with
    objectness) and NMS-free end-to-end heads (1, N, 6).
    """

    kind = "onnx"

    def __init__(self, model, imgsz=416, conf=0.25, iou=0.5, classes=(0,), max_det=50,
                 fmt="auto", names=None, input_name=None, normalize=True,
                 allow_upscale=False):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = input_name or self.sess.get_inputs()[0].name
        # int -> square. tuple -> (height, width), the same order letterbox and
        # ultralytics use, so a wide tile can match a wide tensor instead of being
        # letterboxed into a square one with wasted padding.
        self.imgsz = ((int(imgsz), int(imgsz)) if isinstance(imgsz, int)
                      else (int(imgsz[0]), int(imgsz[1])))
        self.allow_upscale = bool(allow_upscale)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = set(int(c) for c in classes) if classes else None
        self.max_det = int(max_det)
        self.fmt = fmt
        self.normalize = bool(normalize)
        self.names = names or {i: n for i, n in enumerate(COCO_NAMES)}

    def _infer(self, frame):
        img, ratio, pad = letterbox(frame, self.imgsz, auto=False,
                                    allow_upscale=self.allow_upscale)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        blob = np.ascontiguousarray(blob[None])
        if self.normalize:
            blob /= 255.0
        return self.sess.run(None, {self.input_name: blob})[0], ratio, pad

    def _decode(self, out):
        o = np.asarray(out, dtype=np.float32)
        if o.ndim == 3:
            o = o[0]
        if o.ndim != 2:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

        fmt = self.fmt
        if fmt == "auto":
            ncols, nrows = o.shape[-1], o.shape[0]
            # (N, 6) is genuinely ambiguous: it is both a transposed YOLOv8 head
            # with exactly 2 classes and an NMS-free end-to-end head. Disambiguate
            # on the last column: an e2e head carries an integer class id there,
            # a v8 head carries a float class score.
            e2e = False
            if ncols == 6 and nrows > 6:
                col = o[:, 5]
                integral = float(np.mean(np.abs(col - np.round(col)) < 1e-6))
                e2e = integral > 0.9
            if e2e:
                fmt = "e2e"
            elif nrows < ncols:
                fmt = "v8_t"
            elif ncols in (5 + len(COCO_NAMES), 85):
                fmt = "v5"
            else:
                fmt = "v8"

        if fmt == "e2e":
            boxes, scores, cls = o[:, :4], o[:, 4], o[:, 5].astype(int)
            return boxes, scores, cls

        if fmt == "v8_t":
            o = o.T  # (N, 4+nc)
        if fmt == "v5":
            cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
            obj = o[:, 4]
            cls_scores = o[:, 5:]
            cls = cls_scores.argmax(1)
            scores = obj * cls_scores[np.arange(len(o)), cls]
        else:
            cx, cy, w, h = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
            cls_scores = o[:, 4:]
            if cls_scores.shape[1] == 0:
                return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
            cls = cls_scores.argmax(1)
            scores = cls_scores[np.arange(len(o)), cls]

        boxes = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        return boxes, scores, cls

    def detect(self, frame, frame_index=0):
        out, ratio, (pad_x, pad_y) = self._infer(frame)
        boxes, scores, cls = self._decode(out)
        if len(boxes) == 0:
            return []
        keep = scores >= self.conf
        if self.classes is not None:
            keep &= np.isin(cls, list(self.classes))
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        if len(boxes) == 0:
            return []
        if self.fmt != "e2e":
            idx = nms_np(boxes, scores, self.iou)
            boxes, scores, cls = boxes[idx], scores[idx], cls[idx]
        if len(boxes) > self.max_det:
            top = scores.argsort()[::-1][:self.max_det]
            boxes, scores, cls = boxes[top], scores[top], cls[top]
        # undo letterbox
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        return [Detection(xyxy=np.asarray(b, dtype=np.float64), score=float(s), cls_id=int(c),
                          cls_name=self.names.get(int(c), str(int(c))))
                for b, s, c in zip(boxes, scores, cls)]


class DummyDetector(BaseDetector):
    """Replays ground-truth boxes from a JSON file. Used by the smoke test so the
    lock / coast / re-ID logic can be verified with no model and no GPU."""

    kind = "dummy"

    def __init__(self, gt_path, cls_id=0):
        import json
        with open(gt_path) as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "frames" in data:
            data = data["frames"]
        self.frames = data
        self.cls_id = int(cls_id)
        self._last = 0

    def _boxes(self, frame_index):
        if isinstance(self.frames, dict):
            entry = self.frames.get(str(frame_index), [])
        else:
            entry = self.frames[frame_index] if frame_index < len(self.frames) else []
        out = []
        for b in entry:
            if len(b) >= 5:
                out.append((b[:4], b[4]))
            else:
                out.append((b[:4], 1.0))
        return out

    def detect(self, frame, frame_index=0):
        self._last = frame_index
        return [Detection(xyxy=np.asarray(b, dtype=np.float64), score=float(s),
                          cls_id=self.cls_id, cls_name="person")
                for b, s in self._boxes(frame_index)]


def build_detector(cfg, track=False):
    backend = str(cfg.get("backend", "auto")).lower()
    if backend == "dummy":
        return DummyDetector(cfg["dummy_gt"], cls_id=(cfg.get("classes") or [0])[0])
    if backend == "auto":
        try:
            import ultralytics  # noqa: F401
            backend = "ultralytics"
        except Exception:
            backend = "onnx"
    if backend == "ultralytics":
        return UltralyticsDetector(
            model=cfg.get("model", "yolov8n.pt"), imgsz=cfg.get("imgsz", 416),
            conf=cfg.get("conf", 0.25), iou=cfg.get("iou", 0.5),
            classes=cfg.get("classes", [0]), device=cfg.get("device", ""),
            half=cfg.get("half", False), max_det=cfg.get("max_det", 50),
            tracker_cfg=cfg.get("tracker_config"), verbose=cfg.get("verbose", False),
            track=track and cfg.get("name", "botsort") != "none")
    if backend == "onnx":
        return OnnxDetector(
            model=cfg.get("model"), imgsz=cfg.get("imgsz", 416), conf=cfg.get("conf", 0.25),
            iou=cfg.get("iou", 0.5), classes=cfg.get("classes", [0]),
            max_det=cfg.get("max_det", 50), fmt=cfg.get("onnx_format", "auto"),
            normalize=cfg.get("normalize", True))
    raise ValueError(f"unknown detector backend: {backend}")
