import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import sqlite3
import os

# ─── Database Setup ───────────────────────────────────────────────────────────

DB_PATH = r"C:\Retail-Analytics\Retail-Analytics\dwell_events.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dwell_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id      INTEGER,
            zone          TEXT,
            entry_time    TEXT,
            exit_time     TEXT,
            dwell_seconds REAL
        )
    """)
    conn.commit()
    conn.close()

def save_event(event):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO dwell_events (track_id, zone, entry_time, exit_time, dwell_seconds)
        VALUES (?, ?, ?, ?, ?)
    """, (
        event["track_id"],
        event["zone"],
        event["entry_time"],
        event["exit_time"],
        event["dwell_seconds"]
    ))
    conn.commit()
    conn.close()

# Initialize database
init_db()

# ─── Model & Tracker ──────────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=70,
    n_init=10,
    max_cosine_distance=0.225,
    nms_max_overlap=0.5
)

# ─── Zone Definitions ─────────────────────────────────────────────────────────

ZONES = {
    "Entrance": np.array([
        (2, 287), (123, 273), (296, 359), (4, 356), (3, 285)
    ], np.int32),

    "Checkout Counter": np.array([
        (1, 97), (123, 154), (125, 273), (3, 285), (2, 95)
    ], np.int32),

    "Center Aisle": np.array([
        (126, 156), (330, 80), (458, 117), (302, 359)
    ], np.int32),

    "Back Aisle": np.array([
        (0, 1), (259, 0), (305, 88), (126, 156), (3, 95), (1, 3)
    ], np.int32),

    "Right Aisle": np.array([
        (461, 118), (638, 126), (576, 357), (302, 357)
    ], np.int32)
}

ZONE_COLORS = {
    "Entrance":         (0, 255, 0),
    "Checkout Counter": (0, 165, 255),
    "Center Aisle":     (255, 255, 0),
    "Back Aisle":       (255, 0, 255),
    "Right Aisle":      (0, 0, 255)
}

# ─── State ────────────────────────────────────────────────────────────────────

active_dwells = {}
dwell_events = []

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_center(l, t, r, b):
    return (int((l + r) / 2), int((t + b) / 2))

def draw_zones(frame):
    overlay = frame.copy()
    for zone_name, polygon in ZONES.items():
        color = ZONE_COLORS[zone_name]
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(frame, [polygon], True, color, 2)
        M = cv2.moments(polygon.astype(np.float32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, zone_name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

def log_event(event):
    dwell_events.append(event)
    save_event(event)

# ─── Main Loop ────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(r"C:\Retail-Analytics\Retail-Analytics\demo_video2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()

    results = model(frame, classes=[0], verbose=False)

    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.55:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], conf, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    draw_zones(frame)

    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        current_ids.add(track_id)
        l, t, r, b = map(int, track.to_ltrb())
        center = get_center(l, t, r, b)

        current_zone = None
        for zone_name, polygon in ZONES.items():
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                current_zone = zone_name
                break

        if track_id not in active_dwells:
            active_dwells[track_id] = {}

        if current_zone:
            if current_zone not in active_dwells[track_id]:
                active_dwells[track_id][current_zone] = now

        for zone_name in list(active_dwells[track_id].keys()):
            if zone_name != current_zone:
                entry_time = active_dwells[track_id][zone_name]
                dwell_seconds = (now - entry_time).total_seconds()
                if dwell_seconds > 1:
                    log_event({
                        "track_id": track_id,
                        "zone": zone_name,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "dwell_seconds": round(dwell_seconds, 2)
                    })
                del active_dwells[track_id][zone_name]

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        label = f"ID {track_id}"
        if current_zone:
            label += f" | {current_zone}"
        cv2.putText(frame, label, (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.circle(frame, center, 4, (0, 255, 255), -1)

    # Handle lost tracks
    lost_ids = set(active_dwells.keys()) - current_ids
    for track_id in lost_ids:
        for zone_name, entry_time in active_dwells[track_id].items():
            dwell_seconds = (now - entry_time).total_seconds()
            if dwell_seconds > 1:
                log_event({
                    "track_id": track_id,
                    "zone": zone_name,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "dwell_seconds": round(dwell_seconds, 2)
                })
        del active_dwells[track_id]

    cv2.putText(frame, f"Dwell Events: {len(dwell_events)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Retail Zone Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─── Terminal Summary ─────────────────────────────────────────────────────────

print(f"\n--- Dwell Event Summary ---")
print(f"Total events logged: {len(dwell_events)}")
for event in dwell_events[-10:]:
    print(event)