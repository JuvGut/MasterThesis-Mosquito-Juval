# Import necessary libraries
import os

def sync_labels_with_images(images_folder, labels_folder):
    """
    Ensure there is a .txt file for every file in the 'images' folder inside the 'labels' folder.
    If there is no corresponding .txt file, it creates one. It also logs which files were missing a .txt file.
    
    Args:
    - images_folder (str): Path to the folder containing image files.
    - labels_folder (str): Path to the folder where label .txt files are stored.
    
    Returns:
    - None: Files are created/modified directly in the filesystem.
    """
    # Ensure both directories exist
    if not os.path.isdir(images_folder) or not os.path.isdir(labels_folder):
        print("One of the directories does not exist.")
        return
    
    # Get list of image filenames without extensions and list of label files
    image_filenames = [os.path.splitext(file)[0] for file in os.listdir(images_folder) if not file.startswith('.')]
    label_files = [file for file in os.listdir(labels_folder) if file.endswith('.txt')]
    
    # Initialize a list to keep track of images without a corresponding .txt file
    missing_labels = []
    
    # Check each image file for a corresponding .txt file in the labels folder
    for image_filename in image_filenames:
        corresponding_label_file = f"{image_filename}.txt"
        if corresponding_label_file not in label_files:
            # If corresponding .txt file does not exist, create it
            open(os.path.join(labels_folder, corresponding_label_file), 'a').close()
            missing_labels.append(image_filename)
    
    # Write the list of images without a corresponding .txt file to a log file
    with open(os.path.join(os.path.dirname(labels_folder), "missing_labels_list.txt"), "w") as log_file:
        for missing_label in missing_labels:
            log_file.write(f"{missing_label}\n")
    
# To execute the function with specified paths, uncomment the following line:
images = "/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset neg pos/images_only_images_pos_neg"
labels = "/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset neg pos/labels_only_mosquito_pos_neg"
sync_labels_with_images(images, labels)




