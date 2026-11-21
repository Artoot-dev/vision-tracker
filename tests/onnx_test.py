#!/usr/bin/env python3
"""ONNX detector test - real onnxruntime inference, no torch, no YOLO download.

Builds a tiny ONNX graph whose output has the exact YOLOv8 layout
(1, 4+nc, N) with known boxes placed in letterboxed coordinates, then runs the
production OnnxDetector over a synthetic frame and checks the decoded boxes come
back in ORIGINAL image coordinates. That covers the parts most likely to be
wrong: the output-layout heuristics, letterbox undo, confidence filtering, NMS
and clipping.

  python tests/onnx_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vt.detect import OnnxDetector, letterbox, nms_np  # noqa: E402

FRAME_W, FRAME_H = 640, 480
IMGSZ = 416
NC = 2


def build_model(path: Path, anchors: np.ndarray):
    """Graph: output = Constant(anchors) + ReduceSum(images * 0).
    The Mul/ReduceSum chain only exists so the declared input is actually used."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = anchors.shape[1]
    zero = numpy_helper.from_array(np.zeros((), dtype=np.float32), "zero")
    const = numpy_helper.from_array(anchors.astype(np.float32), "anchors")

    nodes = [
        helper.make_node("Mul", ["images", "zero"], ["scaled"]),
        helper.make_node("ReduceSum", ["scaled"], ["tot"], keepdims=0),
        helper.make_node("Add", ["anchors", "tot"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_yolo",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                             [1, 3, IMGSZ, IMGSZ])],
        outputs=[helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                              ["b", "c", "n"])],
        initializer=[zero, const],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main():
    rng = np.random.default_rng(3)
    anchors = np.zeros((1, 4 + NC, 8), dtype=np.float32)
    anchors[0, 4:, :] = 0.01
    # anchor 0: the target, centre (100, 50), size 40x60, class 0 @ 0.90
    anchors[0, :4, 0] = [100, 50, 40, 60]
    anchors[0, 4, 0] = 0.90
    # anchor 1: second object, centre (300, 200), size 80x120, class 0 @ 0.80
    anchors[0, :4, 1] = [300, 200, 80, 120]
    anchors[0, 4, 1] = 0.80
    # anchor 2: near-duplicate of anchor 0 @ 0.70 -> must be suppressed by NMS
    anchors[0, :4, 2] = [102, 52, 40, 60]
    anchors[0, 4, 2] = 0.70
    # anchor 3: class 1 @ 0.95 -> must survive when classes=[0,1]
    anchors[0, :4, 3] = [200, 150, 30, 30]
    anchors[0, 5, 3] = 0.95

    with tempfile.TemporaryDirectory() as td:
        model_path = Path(td) / "tiny.onnx"
        build_model(model_path, anchors)
        frame = rng.integers(0, 255, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)

        det = OnnxDetector(model_path, imgsz=IMGSZ, conf=0.25, iou=0.5, classes=[0, 1],
                           fmt="auto")
        dets = det.detect(frame, 0)
        print(f"decoded {len(dets)} detections")
        for d in dets:
            print(f"   cls={d.cls_id} score={d.score:.2f} bbox={[round(float(v), 1) for v in d.xyxy]}")

        assert len(dets) == 3, f"expected 3 after NMS (dup suppressed), got {len(dets)}"
        cls_ids = sorted(d.cls_id for d in dets)
        assert cls_ids == [0, 0, 1], cls_ids

        # independent expectation: letterbox the frame, map the anchor boxes back
        _img, ratio, (pad_x, pad_y) = letterbox(frame, (IMGSZ, IMGSZ))
        print(f"letterbox ratio={ratio:.4f} pad=({pad_x},{pad_y})")

        def expected(cx, cy, w, h):
            x1 = (cx - w / 2 - pad_x) / ratio
            y1 = (cy - h / 2 - pad_y) / ratio
            x2 = (cx + w / 2 - pad_x) / ratio
            y2 = (cy + h / 2 - pad_y) / ratio
            return np.array([max(x1, 0), max(y1, 0), min(x2, FRAME_W - 1), min(y2, FRAME_H - 1)])

        want = [expected(100, 50, 40, 60), expected(300, 200, 80, 120), expected(200, 150, 30, 30)]
        want.sort(key=lambda b: (b[0], b[1]))
        got = sorted((d.xyxy for d in dets), key=lambda b: (b[0], b[1]))
        for w_, g_ in zip(want, got):
            err = float(np.abs(w_ - g_).max())
            assert err < 1.0, f"box mismatch: want {w_} got {g_} (max err {err:.2f}px)"
        print("onnx: box mapping within 1px of independent expectation")

        # class filtering
        only0 = OnnxDetector(model_path, imgsz=IMGSZ, conf=0.25, iou=0.5, classes=[0]).detect(frame, 0)
        assert all(d.cls_id == 0 for d in only0) and len(only0) == 2, [d.cls_id for d in only0]
        # conf filtering
        strict = OnnxDetector(model_path, imgsz=IMGSZ, conf=0.85, iou=0.5, classes=[0, 1]).detect(frame, 0)
        assert len(strict) == 2, len(strict)

        # the transposed layout (1, N, 4+nc) must decode identically
        build_model(model_path, np.ascontiguousarray(anchors.transpose(0, 2, 1)))
        transposed = OnnxDetector(model_path, imgsz=IMGSZ, conf=0.25, iou=0.5, classes=[0, 1],
                                  fmt="auto").detect(frame, 0)
        det.detect(frame, 0)
        t_boxes = sorted((d.xyxy for d in transposed), key=lambda b: (b[0], b[1]))
        for w_, g_ in zip(want, t_boxes):
            assert float(np.abs(w_ - g_).max()) < 1.0, (w_, g_)
        print("onnx: (1, N, 4+nc) layout decodes identically")

    # NMS sanity on its own
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_np(boxes, scores, 0.5)
    assert list(keep) == [0, 2], keep
    print("onnx: PASS")


if __name__ == "__main__":
    main()
