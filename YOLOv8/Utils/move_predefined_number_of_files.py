import os
import shutil
import random

def move_negative_images(source_dir, dest_dir, num_files):
    # Ensure the destination directory exists, create if it does not
    os.makedirs(dest_dir, exist_ok=True)

    # List all 'neg.*' files in the source directory
    neg_files = [f for f in os.listdir(source_dir) if f.startswith('neg.')]

    # Randomly select the specified number of negative files to move
    selected_files = random.sample(neg_files, min(num_files, len(neg_files)))

    # Move the selected files to the destination directory
    for file in selected_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(dest_dir, file)
        shutil.move(src_path, dst_path)

    print(f"Moved {len(selected_files)} files to {dest_dir}")
    
    # Return the list of moved files
    return selected_files

# Example usage (commented out for the review process):
moved_files = move_negative_images(
            '/home/juval.gutknecht/Mosquito_Detection/data/combined_dataset_243_817/neg', 
            '/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset neg pos/images_only_images_pos_neg', 
            16 # Number of files to move
                                    )