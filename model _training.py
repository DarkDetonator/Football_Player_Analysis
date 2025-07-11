from ultralytics import YOLO

# Load architecture + weights
model = YOLO(r"yolo11n.yaml").load(r"C:\Users\josep\OneDrive\Desktop\ballv3\runs\detect\train15\weights\best.pt")

# Now train on new dataset
model.train(data=r"football-5\data.yaml", epochs=50, imgsz=640)
