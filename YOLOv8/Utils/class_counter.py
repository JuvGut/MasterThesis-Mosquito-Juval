import os
from collections import Counter

def count_images(directory_path):
    # List of common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}
    image_count = 0

    # Iterate over files in the given directory
    for file_name in os.listdir(directory_path):
        # Check if the file extension is one of the common image extensions
        if os.path.splitext(file_name)[1].lower() in image_extensions:
            image_count += 1

    # Output the result
    print("Number of image files:", image_count)

def compile_yolo_labels(directory_path):
    # Initialize a counter for class occurrences and a variable for total samples
    class_counter = Counter()
    total_samples = 0

    # Walk through the directory and read each label file
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):  # Assuming label files are '.txt' files
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                total_samples += len(lines)
                for line in lines:
                    class_id = int(line.split()[0])  # Assuming first item is class ID
                    class_counter[class_id] += 1

    # Number of unique classes
    num_classes = len(class_counter)

    # Output the results
    print("Total samples:", total_samples)
    print("Number of classes:", num_classes)
    print("Samples per class:")
    for class_id, count in class_counter.items():
        print(f"Class ID {class_id}: {count}")




# Example usage (commented out for the review process):
# compile_yolo_labels('/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset/all_labels')


from collections import Counter

def compile_yolo_labels(directory_path):
    # Initialize a counter for class occurrences and a variable for total samples
    class_counter = Counter()
    total_samples = 0
    empty_files = []

    # Walk through the directory and read each label file
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):  # Assuming label files are '.txt' files
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if not lines:  # Check if the file is empty
                    empty_files.append(file_name)
                total_samples += len(lines)
                for line in lines:
                    class_id = int(line.split()[0])  # Assuming first item is class ID
                    class_counter[class_id] += 1

    # Write names of empty .txt files to a file
    if empty_files:
        with open('empty_txt_files.txt', 'w') as ef:
            for name in empty_files:
                ef.write(name + '\n')

    # Number of unique classes and empty files
    num_classes = len(class_counter)
    num_empty_files = len(empty_files)

    # Output the results
    print("Total samples:", total_samples)
    print("Number of classes:", num_classes)
    print("Samples per class:")
    for class_id, count in class_counter.items():
        print(f"Class ID {class_id}: {count}")
    print("Number of empty files:", num_empty_files)


# Example usage (commented out for review):
count_images('/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset neg pos/images_only_images_pos_neg')

# Example usage (commented out for the review process):
compile_yolo_labels('/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset neg pos/labels_only_mosquito_pos_neg')