import cv2
from ultralytics import YOLO

# 1. Load a pre-trained YOLO model (nano version for speed)
model = YOLO("yolov8n.pt") 

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Run detection on the current frame
    results = model(frame)

    # 4. Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # 5. Display the result
    cv2.imshow("Object Detector", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()