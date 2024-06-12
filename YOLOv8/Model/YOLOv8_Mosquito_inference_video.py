import os
from ultralytics import YOLO
import cv2

# Set paths
path = '/home/juval.gutknecht/Mosquito_Detection/Git/YOLO'
VIDEO_PATH = os.path.join(path, 'Mosquito_buenos_aires_social_media.mp4')
OUTPUT_VIDEO_PATH = os.path.join(path, 'analyzed_Mosquito_buenos_aires_social_media.mp4')

model_path = os.path.join(path, 'runs', 'detect', 'train18_yolov8n_with_negatives_1000', 'weights', 'best.pt')

# Load YOLO model
model = YOLO(model_path)
threshold = 0.1

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video file")
    exit()

# Get video properties
H, W, _ = frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            label = f"INSECT: {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the processed frame to the output video
    out.write(frame)
    print("Processed a frame")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to {OUTPUT_VIDEO_PATH}")