import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("demo_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection, person class only (class 0)
    results = model(frame, classes=[0], verbose=False)

    # Annotate frame
    annotated_frame = results[0].plot()

    cv2.imshow("Retail Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()