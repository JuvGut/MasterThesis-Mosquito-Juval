import cv2
import os
from pathlib import Path
from tqdm import tqdm

class GlueTrapCropper:
    def __init__(self, folder_path, crop_region):
        self.folder_path = Path(folder_path)
        self.crop_region = crop_region  # (x, y, width, height)

    def crop_glue_trap(self, image_path):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        # Crop image
        x1, y1, x2, y2 = self.crop_region
        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    def save_cropped_image(self, image, save_path):
        cv2.imwrite(save_path, image)

    def process_folder(self):
        # Ensure the output directory exists
        output_dir_name = f"{self.folder_path.name}_crop"
        output_dir = self.folder_path.parent / output_dir_name
        output_dir.mkdir(exist_ok=True)

        image_paths = sorted(self.folder_path.glob('*.JPG'))

        # Process all JPEG images in the folder
        for image_path in tqdm(image_paths, desc='Cropping images'):
            cropped_image = self.crop_glue_trap(image_path)
            save_path = output_dir / f"{image_path.name}"
            self.save_cropped_image(cropped_image, str(save_path))

# Assuming the crop region is the central part of the image
# These coordinates would need to be adjusted after identifying the exact region
# crop_region = (950, 250, 6800, 3900)  # (x tl, y tl, x br, y br)   
crop_region = (720, 68, 6656, 3876) # camera 243 coordinates: 
# crop_region = (950, 250, 6800, 3900) # camera 817 coordinates: 

# Create an instance of the cropper and process the folder
cropper = GlueTrapCropper(folder_path='/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight', crop_region=crop_region)
cropper.process_folder()