import os
from PIL import Image
from PIL.ExifTags import TAGS

def was_flash_used(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        if exif_data is not None:
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                if decoded == 'Flash':
                    if value % 2 == 0:  # Check if the least significant bit is 1 (flash used) or 0 (flash not used)
                        return True
                    break
    except IOError:
        print(f"Error opening or processing file {image_path}")
    return False

def save_images_with_flash(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            if was_flash_used(file_path):
                destination_path = os.path.join(destination_folder, file_name)
                Image.open(file_path).save(destination_path)
                print(f"Saved: {destination_path}")

# Usage
source_folder = '/home/juval.gutknecht/Mosquito_Detection/data/ATSB_camera_243'
destination_folder = '/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight'
save_images_with_flash(source_folder, destination_folder)