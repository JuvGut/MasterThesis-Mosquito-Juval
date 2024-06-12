import pandas as pd

# Replace 'your_file.xlsx' with the path to your Excel file
excel_file = '/home/juval.gutknecht/Mosquito_Detection/data/ATSB Positive Images/ATSB Positive Image Excel/ATSB Positive Images semi field  230120.xlsx'
df = pd.read_excel(excel_file)

# Example mapping of "Mosquito species" and "Sex" to class integers
# Update this with actual mappings from your data
class_mapping = {
    ('An.arabiensis Male'): 0,
    ('An.arabiensis Female'): 1,
    ('An.funestus Male'): 2,
    ('An.funestus Female'): 3,
    # Add more mappings as needed
}

# Inverse mapping to get class names from IDs for labels.txt
id_to_class = {v: k for k, v in class_mapping.items()}

# Generate YOLO label files
for index, row in df.iterrows():
    image_no = str(row['Image No']).zfill(8)  # Ensure image number is 8 characters
    class_name = f"{row['Mosquito species']} {row['Sex']}"
    object_class = class_mapping[class_name]

    # Placeholder values for bounding box coordinates
    x_center, y_center, width, height = 0.5, 0.5, 0.1, 0.1  # Placeholder values

    # Write the YOLO formatted label to a .txt file
    with open(f"{image_no}.txt", 'w') as label_file:
        label_file.write(f"{object_class} {x_center} {y_center} {width} {height}\n")

# Generate labels.txt containing class names
with open('labels.txt', 'w') as labels_file:
    for class_id in sorted(id_to_class):
        labels_file.write(f"{id_to_class[class_id]}\n")