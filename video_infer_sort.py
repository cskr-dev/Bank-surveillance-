#!/usr/bin/env python3
"""
YOLOv8 + OpenCV Video Inference
- Multiclass confidence aggregation
- Frame-wise object tracking (SORT)
- CPU-only, GB10 safe
"""

import cv2
import json
import joblib
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ================= SORT TRACKER =================
# Minimal SORT implementation (CPU safe)

class SortTracker:
    def __init__(self, max_age=15, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = areaA + areaB - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        updated_tracks = {}
        used = set()

        for tid, tbox in self.tracks.items():
            best_iou, best_det = 0, None
            for i, det in enumerate(detections):
                if i in used:
                    continue
                iou = self._iou(tbox["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou, best_det = iou, i

            if best_iou > self.iou_threshold:
                updated_tracks[tid] = {
                    "bbox": detections[best_det]["bbox"],
                    "cls": detections[best_det]["cls"],
                    "conf": detections[best_det]["conf"],
                    "age": 0
                }
                used.add(best_det)
            else:
                tbox["age"] += 1
                if tbox["age"] <= self.max_age:
                    updated_tracks[tid] = tbox

        for i, det in enumerate(detections):
            if i not in used:
                updated_tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "age": 0
                }
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks


# ================= LOAD MODEL =================

ARTIFACT_DIR = "yolo_cpu_artifacts"
JOBLIB_PATH = f"{ARTIFACT_DIR}/tool_classifier.joblib"
CLASS_MAP_PATH = f"{ARTIFACT_DIR}/metadata/class_id_map.json"

bundle = joblib.load(JOBLIB_PATH)
model = YOLO(bundle["weights"])
model.to("cpu")

with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

id_to_class = {v: k for k, v in class_map.items()}

# ================= VIDEO =================

VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_tracked.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

tracker = SortTracker()

# ================= AGGREGATION =================
global_confidence = defaultdict(list)

# ================= PROCESS =================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        imgsz=bundle["img_size"],
        conf=0.25,
        device="cpu",
        verbose=False
    )

    detections = []
    frame_confidence = defaultdict(list)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cname = id_to_class[cls_id]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "cls": cname,
                "conf": conf
            })

            frame_confidence[cname].append(conf)
            global_confidence[cname].append(conf)

    tracks = tracker.update(detections)

    for tid, t in tracks.items():
        x1, y1, x2, y2 = t["bbox"]
        label = f"{t['cls']} ID:{tid} {t['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    y_offset = 20
    for cname, confs in frame_confidence.items():
        avg_conf = sum(confs) / len(confs)
        cv2.putText(
            frame,
            f"{cname}: {avg_conf:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        y_offset += 20

    writer.write(frame)

cap.release()
writer.release()

# ================= FINAL REPORT =================

print("\n=== GLOBAL CONFIDENCE SUMMARY ===")
for cname, confs in global_confidence.items():
    print(f"{cname}: mean={np.mean(confs):.3f}, max={np.max(confs):.3f}")
