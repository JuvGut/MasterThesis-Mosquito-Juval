import os

def add_a_to_filename(filename):
    """
    Adds the letter 'a' at the end of the filename, before the file extension.
    """
    name, ext = os.path.splitext(filename)
    return f"{name}i{ext}"

def rename_files_in_folder(folder_path):
    """
    Renames all files in the given folder by adding the letter 'a' at the end of the filename.
    """
    for filename in os.listdir(folder_path):
        new_filename = add_a_to_filename(filename)
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

# Example usage
folder_path = "/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv/labels"

rename_files_in_folder(folder_path)