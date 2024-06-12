import cv2 as cv
import numpy as np
import os
import json
import random as rng

class ImageBoundingBoxGenerator:
    def __init__(self, folder_path, save_path, threshold=127, min_area=100, square_size=50):
        self.folder_path = folder_path
        self.save_path = save_path
        self.threshold = threshold
        self.min_area = min_area
        self.square_size = square_size
        self.labels_folder = os.path.join(self.save_path, "labels")  # Folder for YOLO label files
        rng.seed(12345)

    def process_images(self):
        print(f"Processing images in {self.folder_path}")
        if not os.path.exists(self.labels_folder):
            os.makedirs(self.labels_folder)
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png')):
                    print(f"Processing {filename}")
                    image_path = os.path.join(root, filename)
                    image = cv.imread(image_path)
                    if image is not None:
                        bounding_boxes = self.find_bounding_boxes(image)
                        if bounding_boxes:
                            # Limit to max 2 bounding boxes
                            bounding_boxes = bounding_boxes[:1]
                            self.save_yolo_labels(bounding_boxes, filename)
                            self.draw_and_save(image, bounding_boxes, filename)
                        else:
                            print(f"No bounding boxes found for {filename}")
                    else:
                        print(f'Failed to load image: {filename}')

    def find_bounding_boxes(self, image):
        src_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3, 3))
        canny_output = cv.Canny(src_gray, self.threshold, self.threshold * 2)
        contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for c in contours:
            boundRect = cv.boundingRect(c)
            area = boundRect[2] * boundRect[3]
            if area >= self.min_area:
                center_x, center_y = boundRect[0] + boundRect[2] / 2, boundRect[1] + boundRect[3] / 2
                width, height = boundRect[2] / image.shape[1], boundRect[3] / image.shape[0]
                bounding_boxes.append([center_x / image.shape[1], center_y / image.shape[0], width, height])
        # Sort bounding boxes by area in descending order and pick the top 2
        bounding_boxes.sort(key=lambda x: x[2]*x[3], reverse=True)
        return bounding_boxes

    def draw_and_save(self, image, bounding_boxes, filename):
        height, width = image.shape[:2]
        for box in bounding_boxes:
            center_x, center_y, rel_width, rel_height = box
            x1 = int((center_x - rel_width / 2) * width)
            y1 = int((center_y - rel_height / 2) * height)
            x2 = int((center_x + rel_width / 2) * width)
            y2 = int((center_y + rel_height / 2) * height)
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        save_image_path = os.path.join(self.save_path, 'bounded_' + filename)
        cv.imwrite(save_image_path, image)

    def save_yolo_labels(self, bounding_boxes, filename):
        base_filename = os.path.splitext(filename)[0]
        label_path = os.path.join(self.labels_folder, base_filename + '.txt')
        with open(label_path, 'w') as f:
            for box in bounding_boxes:
                class_label = 0  # Assuming a single class label
                f.write(f"{class_label} {' '.join(map(str, box))}\n")

# Uncomment for usage
folder_path = '/home/juval.gutknecht/Mosquito_Detection/data/combined_dataset_243_817_bg_sub/pos'
save_folder = '/home/juval.gutknecht/Mosquito_Detection/Inception-ResNet-v2/old_runs/new_good_Bounding_Boxes/'
threshold = 50
min_area = 5
square_size = 150
generator = ImageBoundingBoxGenerator(folder_path, save_folder, threshold, min_area, square_size)
generator.process_images()
