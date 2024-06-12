# imports
import os
from random import sample, shuffle
from unicodedata import numeric
from venv import create
from sklearn.model_selection import train_test_split
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
from tabulate import tabulate
import numpy as np

# Paths to dataset directories and hyperparameters
datadir_img = "/home/juval.gutknecht/Mosquito_Detection/data/combined_dataset_243_817"
datadir_bg_img = "/home/juval.gutknecht/Mosquito_Detection/data/combined_dataset_243_817_bg_sub"

batch_size = 20
num_workers = 4
input_size = (512, 512)


# # Paths to dataset directories
# traindir = os.path.join(datadir_img)
# traindir_bg = os.path.join(datadir_bg_img)


# (custom) transformations
class MyCustomTransformation:
    def __init__(self, p=0.5):
        self.p = p
        self.resize = v2.Resize(input_size, antialias=True)
        self.transform = v2.Compose([
            v2.ToImage(), # convert to PIL image
            v2.ToDtype(torch.float32, scale=True), # analogous to transforms.ToTensor()

            # augmentations
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.GaussianBlur(3, sigma=(0.1, 1.0)),

        ])
        

    def __call__(self, img):
        img = self.resize(img)
        img = self.transform(img)
        # img = self.adaptive_gauss_noise_per_channel(img) # add noise
        return img

    def adaptive_gauss_noise_per_channel(self, img):
        if torch.rand(1) < self.p:
            for i in range(img.shape[0]):
                channel_mean = img[i].mean()

                noise = torch.randn_like(img[i]) * channel_mean * torch.rand(1).item() * 0.1
                img[i] = img[i] + noise
        return img

# (custom) transformations for test and test dataset
class MyCustomTransform_test(MyCustomTransformation):
    def __init__(self):
        super().__init__()
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

# custom image dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, bg_img_dirs, transform=None):
        self.transform = transform
        self.img_names = []

        for label, img_dir in img_dirs.items():
            bg_img_dir = bg_img_dirs[label]
            for filename in os.listdir(img_dir):
                if filename.endswith('.JPG'):
                    img_path = os.path.join(img_dir, filename)
                    base, ext = os.path.splitext(filename)
                    bg_img_filename = f'{base}_s{ext}'
                    bg_img_path = os.path.join(bg_img_dir, bg_img_filename)
                    self.img_names.append((img_path, bg_img_path, label))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx): 
        img_path, bg_img_path, label = self.img_names[idx]

        img = read_image(img_path)
        bg_sub_img = read_image(bg_img_path)

        if self.transform:
            img = self.transform(img)
            bg_sub_img = self.transform(bg_sub_img)
            
        if torch.equal(img[0, :, :], img[1, :, :]):
            # print(f'Grayscale image: {img_path}')
            combined_img = torch.cat((img[:1, :, :], bg_sub_img[:1, :, :]), dim=0) # img[Channel, Height, Width] with channel: 0, 1, 2
        else:
            # print(f'Color image: {img_path}')
            img_gray = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img_gray = img_gray.unsqueeze(0)
            combined_img = torch.cat((img_gray[:1, :, :], bg_sub_img[:1, :, :]), dim=0)
        
        numeric_label = 1 if label == 'pos' else 0

        return combined_img, numeric_label

def create_split_indices(dataset_size, split_sizes=(0.7, 0.15, 0.15)):
    indices = list(range(dataset_size))
    train_size, valid_size, test_size = split_sizes
    train_indices, temp_indices = train_test_split(indices, train_size=train_size, shuffle=True, random_state=42)
    valid_indices, test_indices = train_test_split(temp_indices, train_size=valid_size/(valid_size + test_size), shuffle=True, random_state=42)
    return train_indices, valid_indices, test_indices

def create_datasets_from_indices(dataset, train_indices, valid_indices, test_indices):
    train_data = Subset(dataset, train_indices)
    valid_data = Subset(dataset, valid_indices)
    test_data = Subset(dataset, test_indices)
    return train_data, valid_data, test_data

def get_weights(dataset):
    label_counts = {0: 0, 1: 0}
    for _ , label in dataset:
            label_counts[label] += 1
    total_count = sum(label_counts.values())
    weights = {label: total_count / max(count, 1) for label, count in label_counts.items()}
    sample_weights = [weights[label] for _ , label in dataset]
    return torch.tensor(sample_weights, dtype=torch.float32) / total_count

# Initialize full dataset
img_dirs = {'pos': os.path.join(datadir_img, 'pos'), 'neg': os.path.join(datadir_img, 'neg')}
bg_img_dirs = {'pos': os.path.join(datadir_bg_img, 'pos'), 'neg': os.path.join(datadir_bg_img, 'neg')}
full_dataset = CustomImageDataset(img_dirs, bg_img_dirs, transform=MyCustomTransformation())

# Create split indices and datasets
train_indices, valid_indices, test_indices = create_split_indices(len(full_dataset))
train_dataset, valid_dataset, test_dataset = create_datasets_from_indices(full_dataset, train_indices, valid_indices, test_indices)

# Create DataLoader for each dataset
sample_weights = get_weights(train_dataset)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# Debug prints to confirm dataset lengths
# print(f"Dataset Lengths: Train: {len(train_dataset)}, Validation: {len(valid_dataset)}, Test: {len(test_dataset)}")

# batch analysis
def batch_analysis(data_loader):
    batch_composition = []
    for batch in data_loader:
        images, labels = batch
        negatives = (labels == 0).sum().item()
        positives = (labels == 1).sum().item()
        batch_composition.append((negatives, positives))
    return batch_composition

def print_batch_analysis(batch_composition):
    table_data =  []
    for i, (negatives, positives) in enumerate(batch_composition):
        total = negatives + positives
        table_data.append([f"Batch {i + 1}", f'{negatives} ({100*(negatives/total):.0f}%)', f'{positives} ({100*(positives/total):.0f}%)' , f'{total}' ])
    
    headers = ["Batch", "Negatives", "Positives", "Total"]
    print(tabulate(table_data, headers, tablefmt="fancy_grid"))

# print_batch_analysis(batch_analysis(train_loader))
# print_batch_analysis(batch_analysis(valid_loader))
# print_batch_analysis(batch_analysis(test_loader))