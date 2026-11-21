import time, warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")
from ultralytics import YOLO

cap = cv2.VideoCapture('out/synthetic.mp4')
frames = []
for _ in range(6):
    ok, f = cap.read()
    frames.append(f)
cap.release()

m = YOLO('yolov8n.pt')
for _ in range(3):
    m.predict(frames[0], imgsz=416, verbose=False, device='cpu')

print('--- YOLOv8n, single 640x480 frame, CPU only (Intel i5-8257U, no GPU) ---')
for sz in (256, 320, 416, 640):
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        m.predict(frames[0], imgsz=sz, verbose=False, device='cpu', conf=0.25)
        ts.append((time.perf_counter() - t) * 1000)
    print(f'  imgsz {sz:4d}: {np.median(ts):7.1f} ms/frame  -> {1000/np.median(ts):5.1f} fps ceiling')

print('--- does BoT-SORT GMC survive a constant-size stream? ---')
r = None
for i, f in enumerate(frames):
    r = m.track(f, imgsz=416, persist=True, tracker='configs/botsort_reid.yaml',
                verbose=False, device='cpu')
print('  ran 6 frames persist=True with no exception; dets on last frame:', len(r[0].boxes))

print('--- detect-every-5 reload cost: predict() call overhead on an unchanged frame ---')
ts = []
for _ in range(10):
    t = time.perf_counter()
    m.predict(frames[0], imgsz=416, verbose=False, device='cpu', conf=0.25)
    ts.append((time.perf_counter() - t) * 1000)
print(f'  mean {np.mean(ts):.1f} ms, min {min(ts):.1f} ms, max {max(ts):.1f} ms')
print(f'  => a 5-frame tick at 30 fps gives a 166.7 ms budget; at 416 that is '
      f'{166.7/np.median(ts):.1f} inferences per tick of headroom')
for sz in (320, 256):
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        m.predict(frames[0], imgsz=sz, verbose=False, device='cpu', conf=0.25)
        ts.append((time.perf_counter() - t) * 1000)
    print(f'  at imgsz {sz}: {np.median(ts):.1f} ms -> {166.7/np.median(ts):.1f} inferences per 5-frame tick')
