from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO(r"C:\Users\josep\OneDrive\Desktop\ballv3\runs\detect\train15\weights\best.pt")

# Load video file
video_path = r"C:\Users\josep\Videos\Captures\(340) PSG STUNS Real Madrid 4-0 _ FIFA Club World Cup Highlights - YouTube - Google Chrome 2025-07-11 09-02-49.mp4"
cap = cv2.VideoCapture(video_path)

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detected.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Save annotated frame to video
    out.write(annotated_frame)

    # Optional: Print progress
    print(f"Processed frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")

cap.release()
out.release()
print("Finished saving output_detected.mp4")
