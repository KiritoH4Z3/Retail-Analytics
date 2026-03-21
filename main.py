import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8n.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=70,
    n_init=10,
    max_cosine_distance=0.218,
    nms_max_overlap=0.5
)

# Open video
cap = cv2.VideoCapture("demo_video2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection, person class only
    results = model(frame, classes=[0], verbose=False)

    # Extract detections for DeepSORT
    # Format: ([left, top, width, height], confidence, class)
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.55:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], conf, 'person'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked persons
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        # Draw bounding box
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

        # Draw ID label
        cv2.putText(
            frame,
            f"ID {track_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Retail Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()