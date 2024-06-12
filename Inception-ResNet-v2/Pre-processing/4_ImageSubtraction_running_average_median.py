import ast
from curses import window
from email.mime import image
import cv2
import numpy as np
import os
from sklearn import base
import torch
import torchvision.io as io
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, directory, window_size = 7):
        self.directory = directory
        self.window_size = window_size
        self.results_dir = self.set_results_dir(directory)

    def load_image_paths(self):
        supported_extensions = {'.jpg', '.jpeg', '.png'}
        image_paths = []
        try:
            image_paths = [os.path.join(self.directory, file) for file in os.listdir(self.directory)
                           if os.path.isfile(os.path.join(self.directory, file))
                           and os.path.splitext(file)[1].lower() in supported_extensions]
            image_paths.sort()
            return image_paths
        except Exception as e:
            print(f"Error occurred while loading image paths: {str(e)}")
            return image_paths

    def load_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            return img
        else:
            print(f"Warning: Failed to load image at {path}")
        return None

    def set_results_dir(self, directory):
        results_dir_name = f"{directory}_bgsub_ws{self.window_size}"
        results_dir = os.path.join(os.path.dirname(directory), results_dir_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir

    def subtract_image(self, base_image, target_image):
        subtracted_image = cv2.subtract(base_image, target_image)
        return subtracted_image
        
    
    def process_images(self, image_paths):
        if not image_paths:
            print(f"No images found in {self.directory}")
            return
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc='Processing images'):
            base_image = self.load_image(image_path)
            if base_image is None:
                continue
            
            window_start = max(0, i - self.window_size // 2)
            window_end = min(len(image_paths), i + self.window_size // 2 + 1)
            
            moving_averages = []

            for j in range(window_start, window_end):
                window_images = self.load_image(image_paths[j])
                if window_images is not None:
                    moving_averages.append(window_images)

                if len(moving_averages) > 1:
                    window_array = np.stack(moving_averages)
                    moving_average = np.median(window_array, axis=0)
                else:
                    moving_average = base_image

            moving_average = moving_average.astype(np.uint8)
            subtracted_image = self.subtract_image(moving_average, base_image)

            original_filename = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(self.results_dir, f'{original_filename}_s.JPG'), subtracted_image)
            
def main(directory, window_size):
    processor = ImageProcessor(directory, window_size)
    image_paths = processor.load_image_paths()
    processor.process_images(image_paths)

if __name__ == '__main__':
    directory = '/home/juval.gutknecht/Mosquito_Detection/data/ATSB_camera_817_crop_3x_resized'
    window_size = 7
    main(directory, window_size)


# The speed increase with the last implementation is significant. 
# The previous implementation took 1 hour and 20 minutes to process 1000 images, 
# while the new implementation takes about 50 minutes to process 1000 images.
# The new implementation is about 38% faster than the previous implementation.
# However there is still room for improvement and I doubt that it works for the fullsized images.