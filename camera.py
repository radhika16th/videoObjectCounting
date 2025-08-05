import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# No console output
model.overrides['verbose'] = False

# webcam, 0 is the default camera
cap = cv2.VideoCapture(0)

# resize the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1020)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    # Reads the next frame from the stream
    ret, frame = cap.read()
    if frame is None:
        continue

    # Detect objects
    results = model(frame, conf=0.10, stream=False)[0] # conf is the min confidence threshold (filtering weak detections)

    count = 0
    # Draw bounding boxes and labels
    if results.boxes is not None:
        # Loops through all the detected boxes, and skips the ones with low confidence
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < 0.10:
                continue  # Skip low-confidence detections

            # Gets the exact coordinates, class ID of the detected object, and human-readable label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            count += 1

            # Draws the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Object count labeled
    cv2.putText(frame, f'Objects: {count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Show the output frame
    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
