import os

def change_class_to_same(folder_path):
    """
    This function changes the class number to 1 in all YOLO label .txt files within a specified folder.
    
    :param folder_path: Path to the folder containing the YOLO label files.
    """
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Open the .txt file and read its lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Skip processing if the file is empty
            if not lines:
                continue

            # Modify the class in each line to 1
            modified_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                # Change the class to 0 (mosquito)
                parts[0] = '0'
                modified_lines.append(' '.join(parts))
            
            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                for line in modified_lines:
                    file.write(f"{line}\n")
    print('All labels were changed to class 0 successfully!')

def add_four_to_class(folder_path):
    """
    This function adds 4 to the class number in all YOLO label .txt files within a specified folder.
    
    :param folder_path: Path to the folder containing the YOLO label files.
    """
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Open the .txt file and read its lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Skip processing if the file is empty
            if not lines:
                continue

            # Modify the class in each line by adding 4
            modified_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                # Add 4 to the class
                parts[0] = str(int(parts[0]) + 4)
                modified_lines.append(' '.join(parts))
            
            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                for line in modified_lines:
                    file.write(f"{line}\n")


def modify_class_binary(folder_path):
    """
    This function modifies the class numbers in all YOLO label .txt files within a specified folder.
    Classes that are not 0 will be changed to 1, while class 0 remains unchanged. Empty files are left unchanged.
    
    :param folder_path: Path to the folder containing the YOLO label files.
    """
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .txt file
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Open the .txt file and read its lines
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Skip processing if the file is empty
            if not lines:
                continue
            
            # Modify the class in each line
            modified_lines = []
            for line in lines:
                parts = line.strip().split(' ')
                # Change class to 1 if it's not 0
                if parts[0] != '0':
                    parts[0] = '1'
                modified_lines.append(' '.join(parts))
            
            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                for line in modified_lines:
                    file.write(f"{line}\n")
    print('All labels were changed to binary labels successfully!')




# class 0: mosquito


# This function will change the class number to 0 in all YOLO label .txt files within a specified folder
# change_class_to_same('/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset/all_labels_only_mosquito')

# This function will add 4 to the class number in all YOLO label .txt files within a specified folder
# add_four_to_class('/path/to/your/folder')

# This function will change the class number to 1 for all classes except class 0 
# modify_class_binary('/home/juval.gutknecht/Mosquito_Detection/data/ATSB combined dataset only pos/all_labels')

