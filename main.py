import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

# Load model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=70,
    n_init=10,
    max_cosine_distance=0.225,
    nms_max_overlap=0.5
)

# Zone definitions from your store layout
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

# Zone colors (BGR)
ZONE_COLORS = {
    "Entrance":         (0, 255, 0),      # Green
    "Checkout Counter": (0, 165, 255),    # Orange
    "Center Aisle":     (255, 255, 0),    # Cyan
    "Back Aisle":       (255, 0, 255),    # Magenta
    "Right Aisle":      (0, 0, 255)       # Red
}

# Dwell tracking
active_dwells = {}
dwell_events = []

cap = cv2.VideoCapture(r"C:\Retail-Analytics\Retail-Analytics\demo_video2.mp4")

def get_center(l, t, r, b):
    return (int((l + r) / 2), int((t + b) / 2))

def draw_zones(frame):
    overlay = frame.copy()
    for zone_name, polygon in ZONES.items():
        color = ZONE_COLORS[zone_name]
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(frame, [polygon], True, color, 2)
        # Zone label at centroid
        M = cv2.moments(polygon.astype(np.float32))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, zone_name, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()

    # Detection
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

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw zones
    draw_zones(frame)

    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        current_ids.add(track_id)
        l, t, r, b = map(int, track.to_ltrb())
        center = get_center(l, t, r, b)

        # Zone detection
        current_zone = None
        for zone_name, polygon in ZONES.items():
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                current_zone = zone_name
                break

        # Dwell logic
        if track_id not in active_dwells:
            active_dwells[track_id] = {}

        if current_zone:
            if current_zone not in active_dwells[track_id]:
                active_dwells[track_id][current_zone] = now

        # Zone exit detection
        for zone_name in list(active_dwells[track_id].keys()):
            if zone_name != current_zone:
                entry_time = active_dwells[track_id][zone_name]
                dwell_seconds = (now - entry_time).total_seconds()
                if dwell_seconds > 1:
                    dwell_events.append({
                        "track_id": track_id,
                        "zone": zone_name,
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "dwell_seconds": round(dwell_seconds, 2)
                    })
                del active_dwells[track_id][zone_name]

        # Draw box and label
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
                dwell_events.append({
                    "track_id": track_id,
                    "zone": zone_name,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "dwell_seconds": round(dwell_seconds, 2)
                })
        del active_dwells[track_id]

    # HUD
    cv2.putText(frame, f"Dwell Events: {len(dwell_events)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Retail Zone Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Terminal summary
print(f"\n--- Dwell Event Summary ---")
print(f"Total events logged: {len(dwell_events)}")
for event in dwell_events[-10:]:
    print(event)