# import
from csv import excel
from math import exp
import os
import pandas as pd
from pyparsing import col
from sklearn.multiclass import OutputCodeClassifier
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataloader_v1 import test_dataset, test_loader, valid_loader, train_loader
from torchvision import models
from inceptionresnetv2_big_input import InceptionResNetV2 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix


class MosquitoDetector:
    def __init__(self, model, model_path, device, threshold):
        self.device = device
        self.threshold = threshold
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader 
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.inference_folder = os.path.join(os.path.dirname(model_path), 'inference')
        os.makedirs(self.inference_folder, exist_ok=True)

        self.loader_names = {
            self.train_loader: "train",
            self.valid_loader: "valid",
            self.test_loader: "test"
        }
    
    def load_model(self, model_path):        
        if hasattr(self.model, 'fc'):
            nr_filters = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(nr_filters, 1).to(self.device)
        elif hasattr(self.model, 'last_linear'):  # Fallback for models that might use this naming
            nr_filters = self.model.last_linear.in_features
            self.model.last_linear = torch.nn.Linear(nr_filters, 1).to(self.device)
        else:
            raise AttributeError("Model does not have 'fc' or 'last_linear' layers")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def inference_with_selection(self, data_loader):
        grad_cam = GradCAM(self.model, layer_name='conv2d_7b', threshold=self.threshold)
        all_samples = []
        all_predictions = []
        all_labels = []

        loader_name = self.loader_names[data_loader]

        self.model.eval()
        with torch.no_grad():
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                predictions = torch.sigmoid(self.model(samples))
                all_samples.extend(samples.cpu().data)
                all_predictions.extend(predictions.cpu().data)
                all_labels.extend(labels.cpu().data)

        all_samples = torch.stack(all_samples)
        all_predictions = torch.stack(all_predictions)
        all_labels = torch.tensor(all_labels)

        positive_indices = (all_predictions >= self.threshold).nonzero(as_tuple=True)[0]
        negative_indices = (all_predictions < self.threshold).nonzero(as_tuple=True)[0]

        num_positives = min(2, len(positive_indices))
        selected_positives = positive_indices[torch.randperm(len(positive_indices))[:num_positives]]
        
        # num_negatives = 2
        # selected_negatives = negative_indices[torch.randperm(len(negative_indices))[:num_negatives]]

        selected_indices = selected_positives
        selected_indices = selected_indices[torch.randperm(len(selected_indices))]

        # Plotting
        num_selected = len(selected_indices)
        cols = 3
        rows = (num_selected + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(12, 3*rows), squeeze=False)

        for i in range(rows * cols):
            ax = axs[i // cols, i % cols]
            ax.axis('off')

        for i, idx in enumerate(selected_indices):
            row = i // cols
            col = i % cols
            ax = axs[row, col]
            ax.axis('on')
            sample = all_samples[idx]
            prediction = all_predictions[idx]
            label = all_labels[idx]
            
            prediction_label = "Positive" if prediction >= self.threshold else "Negative"
            confidence = prediction.item() if prediction_label == "Positive" else 1 - prediction.item()
            true_label = "Positive" if label == 1 else "Negative"
            title = f"Pred: {prediction_label}\nTrue: {true_label}" # f"Pred: {prediction_label}\nConf: {confidence:.2f}\nTrue: {true_label}"

            # Generate Saliency Map
            input_image = sample.unsqueeze(0).to(self.device)
            target_class = 0 if label == 0 else 1
            target_class = torch.tensor([target_class], dtype=torch.float, device=self.device)
            saliency_map = grad_cam.generate_saliency_map(input_image, target_class)
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            axs[row, 0].imshow(sample[0].numpy(), cmap='gray')
            axs[row, 0].set_title(title)
            axs[row, 0].axis('off')

            axs[row, 1].imshow(sample[1].numpy(), cmap='gray')
            axs[row, 1].axis('off')

            axs[row, 2].imshow(saliency_map, cmap='jet', interpolation='nearest')
            axs[row, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.inference_folder, f"selected_samples_{loader_name}.png"))
        plt.show()

    def log_confusion_matrix(self, all_preds, all_labels, loader_name):

        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix of {loader_name}')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        plt.xticks(rotation=45)

        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                percentage = "{:.2%}".format(cm_normalized[i,j])
                text = f'{cm[i,j]}\n{percentage}'
                ax.text(j, i, text, va='center', ha='center', 
                        color='white' if cm_normalized[i, j] > 0.5 else 'black')
        
        plt.tight_layout()

        plt.savefig(os.path.join(self.inference_folder, f'{loader_name}_confusion_matrix.png'))
        plt.close(fig)
        
    def compute_metrics(self, data_loader, loader_name):
        true_labels, predicted_labels, predicted_probabilities = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities >= self.threshold).astype(int)

                true_labels.extend(labels.cpu().numpy())
                predicted_probabilities.extend(probabilities)
                predicted_labels.extend(predictions)

        # Calculate metrics
        
        precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)
        self.log_confusion_matrix(predicted_labels, true_labels, loader_name)


        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels),
            "recall": recall_score(true_labels, predicted_labels),
            "f1": f1_score(true_labels, predicted_labels),
            "roc_auc": roc_auc_score(true_labels, predicted_probabilities),
            "pr_auc": auc(recall, precision)
            }
        
        return metrics

    def save_metrics_to_excel(self, data_loader):
        train_metrics = self.compute_metrics(self.train_loader, "train")
        val_metrics = self.compute_metrics(self.valid_loader, "valid")
        test_metrics = self.compute_metrics(self.test_loader, "test")
        
        metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
        "Train Set":        [train_metrics["accuracy"], train_metrics["precision"], train_metrics["recall"], train_metrics["f1"], train_metrics["roc_auc"], train_metrics["pr_auc"]],
        "Validation Set":   [val_metrics["accuracy"],   val_metrics["precision"],   val_metrics["recall"],  val_metrics["f1"],  val_metrics["roc_auc"],  val_metrics["pr_auc"]],
        "Test Set":         [test_metrics["accuracy"],  test_metrics["precision"],  test_metrics["recall"], test_metrics["f1"], test_metrics["roc_auc"], test_metrics["pr_auc"]]
        }

        df = pd.DataFrame(metrics_data)
        excel_file_path = os.path.join(self.inference_folder, "metrics.xlsx")
        df.to_excel(excel_file_path, index=False)
        print(f"Metrics saved to {excel_file_path}")

class GradCAM:
    def __init__(self, model, layer_name, threshold):
        self.threshold = threshold
        self.device = device
        self.model = model
        self.layer_name = layer_name
        self.gradient = None
        self.activation = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()
            # print(f"Activation Map Shape: {self.activation.shape}") # [batch_size, num_feature_maps, height, width]

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate_saliency_map(self, input_image, target_class=None):
        input_image = input_image.to(self.device)
        output = self.model(input_image)

        if target_class is None:
            predictions = torch.sigmoid(output) >= self.threshold
            target_class = predictions.float()

        target_class = target_class.to(self.device)
        
        self.model.zero_grad()
        # print("Target class:", target_class.item(), "Expected range: 0 to", 2-1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        target_class = target_class.float()
        class_loss = loss_fn(output, target_class.view_as(output))
        class_loss.backward()

        weights = self.gradient.mean(dim=[2, 3], keepdim=True) # axis=(1, 2))
        # print(f"Weights shape: {weights.shape}") # [num_feature_maps, 1, 1, 1]
        saliency_map = torch.mul(self.activation, weights).sum(dim=1, keepdim=True)
        saliency_map = torch.maximum(saliency_map, torch.tensor(0)).squeeze()
        saliency_map = saliency_map / torch.max(saliency_map)
        saliency_map = saliency_map.cpu().numpy()

        return saliency_map

# Initialize the model
experiment_folder = '/home/juval.gutknecht/Mosquito_Detection/Inception-ResNet-v2/runs/0417-1327-dataset-combined'
model_path = f'{experiment_folder}/best_model.pth'
device = "cuda:7" if torch.cuda.is_available() else "cpu"
model = InceptionResNetV2(num_classes = 2)
# model = models.resnet101(num_classes = 2)
threshold = 0.5

# run the classifier
detector = MosquitoDetector(model, model_path, device, threshold)

detector.inference_with_selection(detector.train_loader)
detector.inference_with_selection(detector.test_loader)
detector.inference_with_selection(detector.valid_loader)
detector.save_metrics_to_excel(detector.test_loader)
