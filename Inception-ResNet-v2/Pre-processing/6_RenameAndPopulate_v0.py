from email.mime import image
import os
import shutil
from matplotlib.pylab import f
import pandas as pd
import glob
import random

from sklearn.model_selection import train_test_split
from tqdm import tqdm

class ImageFileProcessor:
    def __init__(self, label_file, images_directory, random_seed=42):
        self.label_file = label_file
        self.images_directory = images_directory
        self.train_dir = os.path.join(images_directory, 'train')
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self.create_train_directory()
    
    def create_train_directory(self):
        try:
            os.makedirs(self.train_dir, exist_ok=True)
            print(f'Created directory or verified train directory: {self.train_dir}')
        except OSError as error:
            print(f'Error creating or verifying train directory: {error}')
    
    def rename_and_move_images(self):
        df = pd.read_excel(self.label_file)
        missing_files = []

        for _, row in df.iterrows():
            try:
                image_number = str(row['Image No.']) # was str(int(row['Image No.']))
                pattern = os.path.join(self.images_directory, f'0{image_number}*.JPG')
                matching_files = glob.glob(pattern.format(_s='?'))
                
                if not matching_files:
                    # print(f'No matching files for {pattern}')
                    missing_files.append(image_number)
                    continue

                for source in matching_files:
                    label = row['Mosquito landing'] # 'NTO landing' or 'Mosquito landing'
                    image_name = os.path.basename(source)
                    new_filename = f"{'pos' if label == 'Positive' else 'neg'}.{image_name}"
                    destination = os.path.join(self.train_dir, new_filename)
                    shutil.copy(source, destination) 
            except ValueError:
                print(f'Skipping with NaN value: {row}')

        if missing_files:
            missing_files_path = os.path.join(self.images_directory, 'missing_files.txt')
            with open(missing_files_path, 'w') as file:
                file.writelines(f'{file_name}\n' for file_name in missing_files)
            print(f'Missing files written to {missing_files_path}')

    def organize_images(self):
        for label in ['pos', 'neg']:
            label_dir = os.path.join(self.train_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)
                
            for img_path in tqdm(glob.glob(os.path.join(self.train_dir, f'{label}.*'))):
                shutil.move(img_path, label_dir)

    def process(self):
        self.rename_and_move_images()
        self.organize_images()


# Usage
random_seed = 42 
labels_file_path = '/home/juval.gutknecht/Mosquito_Detection/data/ATSB_camera_243_labels.xlsx'
images_directory = '/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight_crop_resized_3x_bg_sub_bs7_normalized'

renamer = ImageFileProcessor(labels_file_path, images_directory)
renamer.process()

