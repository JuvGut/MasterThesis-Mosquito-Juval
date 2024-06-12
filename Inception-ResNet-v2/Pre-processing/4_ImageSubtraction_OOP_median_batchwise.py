from email.mime import image
import cv2
import numpy as np
import os
from sympy import residue
import torch
import torchvision.io as io
import torch.nn.functional as F
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, directory, batch_size=7):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device} device')
        self.directory = directory
        self.batch_size = batch_size
        self.results_dir = self.set_results_dir(directory)

    def load_images(self):
        supported_extensions = {'.JPG', '.jpg', '.jpeg', '.png'}
        images = []
        image_paths = []
        try:
            image_paths = [os.path.join(self.directory, file) for file in os.listdir(self.directory)
                           if os.path.isfile(os.path.join(self.directory, file))
                           and os.path.splitext(file)[1].lower() in supported_extensions]
            image_paths.sort()
            images = []
            for path in image_paths:
                img = io.read_image(path).to(self.device)
                if img is not None and (img.shape[0] == 1 or img.shape[0] == 3):
                    images.append(img)
                else:
                    print(f"Warning: Image at {path} has unexpected shape {img.shape}")
            return images, image_paths
        except Exception as e:
            print(f"Error occurred while loading images: {str(e)}")
            return images, image_paths

    def set_results_dir(self, directory):
        results_dir_name = f"{directory}_bg_sub_bs{self.batch_size}"
        results_dir = os.path.join(os.path.dirname(directory), results_dir_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir

    @staticmethod
    def create_median_image(images, device):
        tensor_images = torch.stack(images).to(device)
        median_image = torch.median(tensor_images, dim=0)[0]
        # cv2.imwrite('median_image.jpg', median_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        return median_image

    @staticmethod
    def subtract_image(median_image, image):
        image = image.type(torch.float32)
        median_image = median_image.type(torch.float32)
        subtracted_image = torch.abs(median_image - image)
        subtracted_image = subtracted_image.clamp(0, 255)
        return subtracted_image

    @staticmethod
    def tensor_to_image(tensor):
        return tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def process_images(self, images, image_paths):
        if images is None or not images:
            print("No images to process.")
            return
        for i in tqdm(range(0, len(images), self.batch_size), desc='Processing batches'):
            batch_images = images[i:i+self.batch_size]
            batch_image_paths = image_paths[i:i+self.batch_size]
            median_image = self.create_median_image(batch_images, self.device)
            for j, image in enumerate(batch_images):
                subtracted_image = self.subtract_image(median_image, image)
                original_filename = os.path.splitext(os.path.basename(batch_image_paths[j]))[0]
                cv2.imwrite(os.path.join(self.results_dir, f'{original_filename}_s.JPG'), self.tensor_to_image(subtracted_image))
            
def main(directory, batch_size):
    processor = ImageProcessor(directory, batch_size)
    images, image_paths = processor.load_images()
    if images is not None:
        images = [image.to(processor.device) for image in images]
    processor.process_images(images, image_paths)

if __name__ == '__main__':
    directory = '/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight_crop_resized_3x'
    batch_size = 7
    main(directory, batch_size)
