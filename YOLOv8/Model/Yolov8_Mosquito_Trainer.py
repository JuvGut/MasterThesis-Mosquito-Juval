from ultralytics import YOLO
import os
import yaml

# Load a model

def load_model(model_path='yolov8n.yaml', pretrained_weights=None):
    """
    Load the YOLO model.
    :param model_path: Path to the model configuration file (YAML) or pretrained model.
    :param pretrained_weights: Path to the pretrained weights, if any.
    :return: Loaded YOLO model.
    """
    if pretrained_weights:
        model = YOLO(model_path).load(pretrained_weights)
    else:
        model = YOLO(model_path)
    return model

# Define the function to generate the data config file
def generate_data_config(img_folder, labels_folder, config_path='data.yaml'):
    """
    Generate a data configuration file for training.
    :param img_folder: Path to the folder containing the images.
    :param labels_folder: Path to the folder containing the labels.
    :param config_path: Path where the data configuration file will be saved.
    """
    data_config = {
        'train': img_folder,
        'val': img_folder,  # Assuming the same folder for training and validation for simplicity
        'test': None,
        'nc': 4,  # Number of classes; adjust based on your dataset
        'names': ['An.arabiensis Male', 
                  'An.arabiensis Female', 
                  'An.funestus Male',
                  'An.funestus Female'
                  ],  # Class names
    }
    with open(config_path, 'w') as config_file:
        yaml.dump(data_config, config_file, default_flow_style=False)


# Function to train the model
def train_model(model, data_config='data.yaml', epochs=100, img_size=640, batch=16, workers=8, device='2,3,4'):
    """
    Train the YOLO model with corrected argument names.
    :param model: Loaded YOLO model.
    :param data_config: Path to the data configuration file.
    :param epochs: Number of training epochs.
    :param img_size: Input image size for the model.
    :param batch: Batch size for training, corrected from 'batch_size' to 'batch'.
    :param workers: Number of workers for data loading.
    :return: Training results.
    """
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch, workers=workers, device=device)
    return results

# Function to evaluate the model
img_folder = '/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv/images'
labels_folder = '/home/juval.gutknecht/Mosquito_Detection/data/yolo_insect_detect_and_inv/labels'
model = load_model()

# generate_data_config(img_folder, labels_folder)
train_model(model, 
            data_config='data_mosquito_pos_inverted.yaml', 
            epochs=1000, 
            img_size=(1000,750), 
            batch=12, 
            workers=8,
            device='1'
            )
