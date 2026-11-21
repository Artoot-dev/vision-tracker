#!/usr/bin/env python3
"""INT8 export + quantisation.

The detector is the only heavy thing on the device, so this is where the edge
budget is won. Order of preference for an edge box:

  tflite int8 (ultralytics export)   best on RPi/Android/CPU-only NPU stacks
  onnx int8 (static, calibrated)     best on x86/ARM with onnxruntime
  ncnn / openvino                    if the target vendor needs them
  onnx fp32                          fallback

Static quantisation needs representative data - use YOUR OWN clip, not COCO, so
the calibration distribution matches the deployment scene.

  # .pt -> INT8 onnx calibrated on the clip
  python tools/export_int8.py --model yolov8n.pt --to onnx-int8 \
      --calib-video myclip.mp4 --imgsz 320 --out models/

  # .pt -> INT8 tflite (no calibration needed, ultralytics handles it)
  python tools/export_int8.py --model yolov8n.pt --to tflite-int8 --imgsz 320
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def letterbox(img, new_shape, color=(114, 114, 114), stride=32):
    """Must match vt.detect.letterbox (auto=False) so calibration statistics come
    from the same input distribution the device will feed."""
    import cv2
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def sample_frames(video, n, size):
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(total // max(n, 1), 1)
    out = []
    i = 0
    while len(out) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        out.append(letterbox(frame, size))
        i += step
    cap.release()
    if not out:
        raise SystemExit(f"could not read frames from {video}")
    return out


def build_calib_reader(frames, input_name):
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(CalibrationDataReader):
        def __init__(self, frames, input_name):
            self.frames = frames
            self.input_name = input_name
            self.i = 0

        def get_next(self):
            if self.i >= len(self.frames):
                return None
            f = self.frames[self.i]
            self.i += 1
            blob = f[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)[None] / 255.0
            return {self.input_name: np.ascontiguousarray(blob)}

        def rewind(self):
            self.i = 0

    return Reader(frames, input_name)


def export_ultralytics(model, fmt, imgsz, int8=False, half=False, data=None):
    from ultralytics import YOLO
    y = YOLO(model)
    kw = {"format": fmt, "imgsz": imgsz}
    if fmt in ("onnx", "tflite", "openvino", "ncnn", "engine"):
        kw["int8"] = bool(int8)
        kw["half"] = bool(half)
        kw["dynamic"] = False
    if fmt == "onnx":
        kw["simplify"] = True
        kw["opset"] = 12
    if data:
        kw["data"] = data
    path = y.export(**kw)
    print(f"exported -> {path}")
    return path


def quantize_onnx(src, dst, frames, per_channel=True, reduce_range=False):
    from onnxruntime.quantization import QuantType, quantize_static
    import onnxruntime as ort

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = build_calib_reader(frames, input_name)
    print(f"quantising {src} -> {dst}  ({len(frames)} calibration frames, "
          f"input '{input_name}')")
    quantize_static(str(src), str(dst), reader,
                    quant_format=__import__("onnxruntime.quantization",
                                            fromlist=["QuantFormat"]).QuantFormat.QDQ,
                    per_channel=per_channel, reduce_range=reduce_range,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8)
    return dst


def size_mb(p):
    return Path(p).stat().st_size / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt (or an existing .onnx for re-quantising)")
    ap.add_argument("--to", required=True,
                    choices=["onnx", "onnx-int8", "tflite-int8", "tflite", "ncnn", "openvino"])
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", default="models")
    ap.add_argument("--calib-video", default=None, help="required for onnx-int8")
    ap.add_argument("--n-calib", type=int, default=120)
    ap.add_argument("--no-per-channel", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(a.model)
    print(f"source {src}  ({size_mb(src):.1f} MB)")

    if a.to in ("onnx", "onnx-int8"):
        base = src if src.suffix == ".onnx" else Path(export_ultralytics(
            str(src), "onnx", a.imgsz, int8=False))
        if a.to == "onnx":
            print(f"done -> {base}  ({size_mb(base):.1f} MB)")
            return
        if not a.calib_video:
            raise SystemExit("--calib-video is required for onnx-int8 "
                             "(use your deployment clip for representative stats)")
        frames = sample_frames(a.calib_video, a.n_calib, (a.imgsz, a.imgsz))
        dst = out_dir / f"{src.stem}_int8.onnx"
        quantize_onnx(base, dst, frames, per_channel=not a.no_per_channel)
        print(f"done -> {dst}  ({size_mb(dst):.1f} MB, fp32 was {size_mb(base):.1f} MB)")
        return

    fmt = {"tflite-int8": ("tflite", True), "tflite": ("tflite", False),
           "ncnn": ("ncnn", False), "openvino": ("openvino", False)}[a.to]
    path = export_ultralytics(str(src), fmt[0], a.imgsz, int8=fmt[1])
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
