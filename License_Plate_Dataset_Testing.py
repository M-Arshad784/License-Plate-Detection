import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction
results = model.predict(
    source="2.jpg",
    conf=0.5,
    save=True
)

# Get the first result and draw detections
img = results[0].plot()

# Display the result
cv2.imshow("License Plate Detected", img)

cv2.waitKey(0)
cv2.destroyAllWindows()