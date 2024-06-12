import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def find_global_min_max(folder_path):
    global_min = np.inf
    global_max = -np.inf
    
    # Ensure to use a tuple for file extensions
    supported_extensions = ('.JPG', '.jpg', '.jpeg', '.png')
    
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        # Check if it's a file and not a directory
        if os.path.isfile(image_path) and file_name.endswith(supported_extensions):
            try:
                image = Image.open(image_path)
                image_array = np.asarray(image).astype(np.float32)
                
                local_min = np.min(image_array)
                local_max = np.max(image_array)
                
                if local_min < global_min:
                    global_min = local_min
                if local_max > global_max:
                    global_max = local_max
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                
    print(f"Global min: {global_min}, Global max: {global_max}")
    return global_min, global_max

def normalize_image_with_global_values(image_path, global_min, global_max):
    try:
        image = Image.open(image_path)
        image_array = np.asarray(image).astype(np.float32)
        
        normalized_array = (image_array - global_min) / (global_max - global_min)
        normalized_image = Image.fromarray((normalized_array * 255).astype(np.uint8))
        # print(normalized_array) # Check if values seem to be normalized

        return normalized_image
    except Exception as e:
        print(f"Error normalizing image {os.path.basename(image_path)}: {e}")
        return None

def normalize_images_in_folder_with_global_values(folder_path):
    normalized_folder_path = f"{folder_path}_normalized"
    
    if not os.path.exists(normalized_folder_path):
        os.makedirs(normalized_folder_path)
    
    global_min, global_max = find_global_min_max(folder_path)
    
    supported_extensions = ('.JPG', '.jpg', '.jpeg', '.png')
    files =[f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(supported_extensions)]
    for file_name in tqdm(files, desc="Normalizing images"):
        image_path = os.path.join(folder_path, file_name)
        
        normalized_image = normalize_image_with_global_values(image_path, global_min, global_max)
        if normalized_image is not None:
            normalized_image_path = os.path.join(normalized_folder_path, file_name)
            normalized_image.save(normalized_image_path)
    
    print(f"All images have been normalized using global min ({global_min}) and max ({global_max}) values and saved in {normalized_folder_path}.")

# Example usage:
path_to_folder = '/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight_crop_resized_3x'
normalize_images_in_folder_with_global_values(path_to_folder)
