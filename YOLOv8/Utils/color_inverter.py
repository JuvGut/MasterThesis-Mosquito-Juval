import os
from PIL import Image, ImageOps

def invert_image_colors(input_path, output_path):
    try:
        # Open an image file
        with Image.open(input_path) as img:
            # Invert the image colors
            inverted_image = ImageOps.invert(img)
            # Save the inverted image
            inverted_image.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def invert_images_in_folder(input_folder):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f"The folder {input_folder} does not exist.")
        return
    
    # Create the output folder
    output_folder = f"{input_folder}_negative"
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # Only process files with image extensions
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Split the filename and extension
            base, ext = os.path.splitext(filename)
            # Construct the new filename with an additional "i"
            new_filename = f"{base}i{ext}"
            output_path = os.path.join(output_folder, new_filename)
            # Invert the image colors and save it with the new filename
            invert_image_colors(input_path, output_path)
        else:
            print(f"Skipping non-image file: {filename}")

# Example usage:
invert_images_in_folder('/home/juval.gutknecht/Mosquito_Detection/data/mosquito detection dataset/all_positive_images')