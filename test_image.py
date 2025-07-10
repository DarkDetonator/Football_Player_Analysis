from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model (replace with your path if needed)
#model = YOLO(r"C:\Users\josep\Downloads\runs\runs\detect\transfer_aug\weights\best.pt")
model = YOLO(r"C:\Users\josep\Downloads\football_best.pt")
# Load your image (replace with your image path)
image_path = r"C:\Users\josep\OneDrive\Pictures\Screenshots\Screenshot 2025-07-09 231131.png"
results = model(image_path)

# Show results with bounding boxes
results[0].show()  # this opens a window with the image (or use .save() to save it)

# Optional: Save the result image
results[0].save(filename="predicted_image.jpg")
