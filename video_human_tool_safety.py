#!/usr/bin/env python3
"""
Human ↔ Tool Association + Safety Rules
CPU-only | Open-source YOLO | SORT tracking
"""

import cv2
import json
import joblib
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ===================== CONFIG =====================

DANGEROUS_TOOLS = {
    "grinder": 0.60,
    "knife": 0.55,
    "chainsaw": 0.60,
    "drill": 0.50
}

ALERT_FRAMES_REQUIRED = 5   # anti-flicker
IOU_ASSOC_THRESHOLD = 0.10

# =================================================


# ================= SORT TRACKER ==================

class SortTracker:
    def __init__(self, max_age=20, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def _iou(self, a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        union = areaA + areaB - inter
        return inter / union if union else 0

    def update(self, detections):
        updated, used = {}, set()

        for tid, t in self.tracks.items():
            best, idx = 0, None
            for i, d in enumerate(detections):
                if i in used:
                    continue
                iou = self._iou(t["bbox"], d["bbox"])
                if iou > best:
                    best, idx = iou, i
            if best > self.iou_threshold:
                updated[tid] = {**detections[idx], "age": 0}
                used.add(idx)
            else:
                t["age"] += 1
                if t["age"] <= self.max_age:
                    updated[tid] = t

        for i, d in enumerate(detections):
            if i not in used:
                updated[self.next_id] = {**d, "age": 0}
                self.next_id += 1

        self.tracks = updated
        return updated

# =================================================


def center_inside(box_small, box_big):
    cx = (box_small[0] + box_small[2]) // 2
    cy = (box_small[1] + box_small[3]) // 2
    return (box_big[0] <= cx <= box_big[2] and
            box_big[1] <= cy <= box_big[3])


# ================= LOAD MODELS ===================

human_model = YOLO("yolov8n.pt").to("cpu")

bundle = joblib.load("yolo_cpu_artifacts/tool_classifier.joblib")
tool_model = YOLO(bundle["weights"]).to("cpu")

with open("yolo_cpu_artifacts/metadata/class_id_map.json") as f:
    class_map = json.load(f)

id_to_tool = {v: k for k, v in class_map.items()}

# ================= VIDEO =========================

cap = cv2.VideoCapture("input_video.mp4")
w, h = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(5))

out = cv2.VideoWriter(
    "output_safety.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

tracker = SortTracker()

# ================= STATE =========================

alert_counter = defaultdict(int)

# ================= PROCESS =======================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    # -------- HUMAN DETECTION --------
    for r in human_model.predict(frame, conf=0.4, device="cpu", verbose=False):
        for box in r.boxes or []:
            if int(box.cls[0]) == 0:
                detections.append({
                    "bbox": list(map(int, box.xyxy[0])),
                    "cls": "HUMAN",
                    "conf": float(box.conf[0])
                })

    # -------- TOOL DETECTION --------
    for r in tool_model.predict(frame, conf=0.25, device="cpu", verbose=False):
        for box in r.boxes or []:
            detections.append({
                "bbox": list(map(int, box.xyxy[0])),
                "cls": id_to_tool[int(box.cls[0])],
                "conf": float(box.conf[0])
            })

    tracks = tracker.update(detections)

    humans = {tid: t for tid, t in tracks.items() if t["cls"] == "HUMAN"}
    tools = {tid: t for tid, t in tracks.items() if t["cls"] != "HUMAN"}

    associations = {}

    # -------- ASSOCIATION LOGIC --------
    for hid, h in humans.items():
        best_tool, best_score = None, 0
        for tid, t in tools.items():
            iou = tracker._iou(h["bbox"], t["bbox"])
            if iou > IOU_ASSOC_THRESHOLD and center_inside(t["bbox"], h["bbox"]):
                if iou > best_score:
                    best_tool, best_score = tid, iou
        if best_tool is not None:
            associations[hid] = tools[best_tool]

    # -------- DRAW + SAFETY --------
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t["bbox"]
        color = (0, 255, 0) if t["cls"] == "HUMAN" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{t['cls']} ID:{tid}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for hid, tool in associations.items():
        tool_name = tool["cls"]
        conf = tool["conf"]

        if tool_name in DANGEROUS_TOOLS and conf >= DANGEROUS_TOOLS[tool_name]:
            alert_counter[(hid, tool_name)] += 1
            if alert_counter[(hid, tool_name)] >= ALERT_FRAMES_REQUIRED:
                cv2.putText(frame,
                            f"⚠ ALERT: HUMAN {hid} USING {tool_name.upper()}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3)
                print(f"[ALERT] Human {hid} holding {tool_name}")

    out.write(frame)

cap.release()
out.release()

print("\n✅ SAFETY ANALYSIS COMPLETED")
