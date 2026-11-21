"""Interactive subject selection on the detection frame.

GUI mode (default): OpenCV window, click the box you want to track.
Headless mode: --point x,y or --roi x,y,w,h or --auto, so the same pipeline runs
over SSH / on the device with no display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .geometry import box_area, box_center, iou_xyxy, point_in_box


def _label(frame, i, det, color=(0, 255, 255), thick=2):
    x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    name = det.cls_name or str(det.cls_id)
    txt = f"[{i}] {name} {det.score:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 3, max(y1 - 5, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def _auto_pick(dets, rule="largest"):
    if not dets:
        return None
    if rule == "conf":
        return int(np.argmax([d.score for d in dets]))
    if rule == "center":
        areas = np.array([box_area(d.xyxy) for d in dets])
        return int(np.argmax(areas))
    return int(np.argmax([box_area(d.xyxy) for d in dets]))


def pick_headless(dets, mode="auto", point=None, roi=None, rule="largest"):
    if not dets:
        return None
    if mode == "point" and point is not None:
        hits = [i for i, d in enumerate(dets) if point_in_box(point, d.xyxy)]
        if not hits:
            return None
        return min(hits, key=lambda i: box_area(dets[i].xyxy))
    if mode == "roi" and roi is not None:
        x, y, w, h = [float(v) for v in roi]
        target = np.array([x, y, x + w, y + h], dtype=np.float64)
        ious = [iou_xyxy(target, d.xyxy) for d in dets]
        return int(np.argmax(ious))
    return _auto_pick(dets, rule)


def select_interactive(frame, dets, window="select subject", save_path=None,
                       auto_rule="largest"):
    """Returns the chosen index, or None if the user aborted."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for GUI selection")
    state = {"idx": None, "hover": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        hits = [i for i, d in enumerate(dets) if point_in_box((x, y), d.xyxy)]
        if hits:
            # smallest containing box wins, so a nested object is selectable
            state["idx"] = min(hits, key=lambda i: box_area(dets[i].xyxy))

    def on_track(event, x, y, flags, param):
        hits = [i for i, d in enumerate(dets) if point_in_box((x, y), d.xyxy)]
        state["hover"] = min(hits, key=lambda i: box_area(dets[i].xyxy)) if hits else None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        cv2.setMouseCallback(window, on_track)
    except Exception:
        pass

    print("[select] click the object to track | 0-9 pick by index | a=largest | "
          "c=highest conf | space/enter=confirm | q=abort")
    while True:
        canvas = frame.copy()
        for i, d in enumerate(dets):
            sel = (state["idx"] == i)
            hov = (state["hover"] == i)
            color = (0, 255, 0) if sel else ((255, 200, 0) if hov else (0, 255, 255))
            _label(canvas, i, d, color, 3 if sel else 2)
        tip = "click a box  |  a=auto  c=conf  space=confirm  q=abort"
        cv2.putText(canvas, tip, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(canvas, tip, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if state["idx"] is not None:
            s = f"SELECTED [{state['idx']}]"
            cv2.putText(canvas, s, (8, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, s, (8, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10, 32):  # enter / space
            if state["idx"] is not None:
                break
        elif key == ord("a"):
            state["idx"] = _auto_pick(dets, auto_rule)
        elif key == ord("c"):
            state["idx"] = _auto_pick(dets, "conf")
        elif key in (ord("q"), 27):
            state["idx"] = None
            break
        elif ord("0") <= key <= ord("9"):
            i = key - ord("0")
            if i < len(dets):
                state["idx"] = i
    cv2.destroyWindow(window)
    if save_path and state["idx"] is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)
    return state["idx"]


def resolve_selection(cfg, frame, dets, save_path=None):
    """cfg = selection config block. Returns (index, detail dict) or (None, detail)."""
    mode = str(cfg.get("mode", "click")).lower()
    detail = {"mode": mode}
    if not dets:
        detail["reason"] = "no detections on the selection frame"
        return None, detail
    if mode == "click":
        try:
            idx = select_interactive(frame, dets, save_path=save_path,
                                     auto_rule=cfg.get("auto_rule", "largest"))
        except Exception as exc:
            # headless box, no display, or a broken GUI build: do not lose the run
            idx = _auto_pick(dets, cfg.get("auto_rule", "largest"))
            detail["gui_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[select] interactive selection unavailable ({type(exc).__name__}); "
                  f"falling back to auto pick. Use --select point --point x,y to choose "
                  f"explicitly, or --select roi x,y,w,h.")
    else:
        idx = pick_headless(dets, mode=mode, point=cfg.get("point"), roi=cfg.get("roi"),
                            rule=cfg.get("auto_rule", "largest"))
    if idx is None:
        detail["reason"] = "no subject selected"
        detail["detections"] = [[float(v) for v in d.xyxy] for d in dets]
        return None, detail
    detail["index"] = int(idx)
    detail["bbox"] = [float(v) for v in dets[idx].xyxy]
    detail["cls_name"] = dets[idx].cls_name
    detail["score"] = float(dets[idx].score)
    if save_path and cv2 is not None:
        canvas = frame.copy()
        for i, d in enumerate(dets):
            _label(canvas, i, d, (0, 255, 0) if i == idx else (0, 255, 255),
                   3 if i == idx else 1)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), canvas)
        detail["frame_png"] = str(save_path)
    return int(idx), detail
