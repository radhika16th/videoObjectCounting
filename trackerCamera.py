import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load YOLOv8 model
model = YOLO('yolov8l.pt')

# No console output
model.overrides['verbose'] = False

# webcam, 0 is the default camera
cap = cv2.VideoCapture(0)

# resize the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1020)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# line
line_x = 510 # change this value to adjust the vertical line position

total_count = 0
track_history = {}

while True:
    # Reads the next frame from the stream
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect objects
    results = model.track(frame, persist=True, conf=0.10)[0] # conf is the min confidence threshold (filtering weak detections)

    # Draw bounding boxes and labels
    if results.boxes.id is not None: # type: ignore[attr-defined]

        # Extracts the track IDs and bounding boxes
        ids = results.boxes.id.cpu().numpy().astype(int) # type: ignore[attr-defined]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int) # type: ignore[attr-defined]

        # Iterate through each detected object
        for track_id, box in zip(ids, boxes):
            # Get the center x-coordinate of the bounding box
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)

            # Check if the center x-coordinate crosses the vertical line
            prev_cx = track_history.get(track_id, None)
            track_history[track_id] = cx

            # Count logic
            if prev_cx is not None:
                if prev_cx < line_x <= cx:
                    total_count += 1
                elif prev_cx > line_x >= cx and total_count > 0:
                    total_count -= 1

            # Draws the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw vertical counting line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

    # Show total object count
    cv2.putText(frame, f'Count: {total_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Show the output frame
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()