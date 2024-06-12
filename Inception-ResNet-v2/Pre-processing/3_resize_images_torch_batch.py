import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB') # Ensure RGB even for grayscale images
        if self.transform:
            image = self.transform(image)
        return image, img_name


def resize_images(input_dir, output_dir, resize_factor, device, batch_size):
    # Check if GPU is used
    print(f"Using device: {device}") 

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define transformation: Resize and then convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Use the ImageFolder class or a custom dataset for loading images
    dataset = CustomImageDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Process and save each image
    for images, filenames in tqdm(dataloader, desc="Resizing Images:"):
        # Transfer to GPU if available
        images = images.to(device)
        
        # calculate new size
        _, _, height, width = images.shape
        new_height = height // resize_factor
        new_width = width // resize_factor
        
        # Resize the images
        resized_images = torch.nn.functional.interpolate(images, size=(new_height, new_width), mode='bicubic', align_corners=False)

        # Save each resized image from the batch individually
        for image, filename in zip(resized_images, filenames):
            save_path = os.path.join(output_dir, filename)
            utils.save_image(image, save_path)


    # Check if all the resized images are saved
    if len(os.listdir(input_dir)) == len(os.listdir(output_dir)):
        print("All images were resized and saved successfully")
    else:
        print("Not all images were resized.")


# Define the input and output directories
input_directory = "/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight_crop"
output_directory = "/home/juval.gutknecht/Mosquito_Detection/data/onlyDaylight_crop_resized_3x"
scale_factor = 3        # 1/x of the original size
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 32

# Resize the images
resize_images(input_directory, output_directory, scale_factor, device, batch_size)