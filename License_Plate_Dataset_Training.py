from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="D:/CV/License Plate Detection/data.yaml",
    epochs=50,
    batch=8
)