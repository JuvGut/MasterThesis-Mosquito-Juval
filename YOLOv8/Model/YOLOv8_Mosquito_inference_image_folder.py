import os

from ultralytics import YOLO
import cv2

path = '/home/juval.gutknecht/Mosquito_Detection/Git/YOLO'
# VIDEOS_DIR = os.path.join('.', 'videos')

IMAGES_DIR = os.path.join(path, '/home/juval.gutknecht/Mosquito_Detection/Git/YOLO/spider_imgs_fullsize')
OUTPUT_DIR = os.path.join(path, 'processed_images_allpos')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# video_path = os.path.join(VIDEOS_DIR, 'alpaca1.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join(path, '.', 'runs', 'detect', 'train12_yolov8m_only_mosquito_with_negatives_1000', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.3

for filename in os.listdir(IMAGES_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
        image_path = os.path.join(IMAGES_DIR, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Process the image
        results = model(image)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                # label = f"{results.names[int(class_id)].upper()}: {score:.2f}"
                label = f"INSECT: {score:.2f}"
                cv2.putText(image, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Save the processed image
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, image)
        print(f"Processed image saved to {output_path}")

cv2.destroyAllWindows()