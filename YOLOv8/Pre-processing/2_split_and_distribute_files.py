import os
import shutil
from sklearn.model_selection import train_test_split

# Define the directory paths
src_images_dir = '/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv/images_all'
src_labels_dir = '/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv/labels_all'
dest_dir_base = '/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv'

# Define the destination directories for each set
dest_dirs = {
    'train': ('images/train', 'labels/train'),
    'val': ('images/val', 'labels/val'),
    'test': ('images/test', 'labels/test')
}

def split_and_distribute_files(src_images_dir, src_labels_dir, dest_dir_base, dest_dirs):
    # List all image files
    image_files = [f for f in os.listdir(src_images_dir) if os.path.isfile(os.path.join(src_images_dir, f))]
    
    # Shuffle and split the file names into training, validation, and test sets
    train_files, test_files = train_test_split(image_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    
    # Create a dictionary to map file lists to their respective directories
    file_lists = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Iterate over each set and copy the files to their new destination
    for set_name, file_list in file_lists.items():
        img_dest_dir, label_dest_dir = dest_dirs[set_name]
        # Ensure the destination directories exist
        os.makedirs(os.path.join(dest_dir_base, img_dest_dir), exist_ok=True)
        os.makedirs(os.path.join(dest_dir_base, label_dest_dir), exist_ok=True)
        
        # Copy each file
        for filename in file_list:
            # Determine source and destination paths for images and labels
            src_img_path = os.path.join(src_images_dir, filename)
            dest_img_path = os.path.join(dest_dir_base, img_dest_dir, filename)
            label_filename = filename.rsplit('.', 1)[0] + '.txt'
            src_label_path = os.path.join(src_labels_dir, label_filename)
            dest_label_path = os.path.join(dest_dir_base, label_dest_dir, label_filename)
            
            # print(f"Copying image {src_img_path} to {dest_img_path}")
            # print(f"Copying label {src_label_path} to {dest_label_path}")
            
            # Copy the files
            shutil.copy2(src_img_path, dest_img_path)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dest_label_path)
            else:
                print(f"Warning: No label file found for {src_label_path}")
    print("Files have been distributed successfully!")

# Commenting out the function call to adhere to instructions
split_and_distribute_files(src_images_dir, src_labels_dir, dest_dir_base, dest_dirs)
