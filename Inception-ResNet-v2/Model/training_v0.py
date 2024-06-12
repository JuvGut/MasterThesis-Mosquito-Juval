# imports
from math import log
import os
import datetime
from tabnanny import check
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from sympy import true
from tqdm import tqdm
from dataloader_v1 import train_loader, valid_loader
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, auc, f1_score, roc_auc_score
from torchvision import models, utils
from inceptionresnetv2_big_input import InceptionResNetV2  # Adjust as necessary for your model
import torch
import torch.nn as nn
from torch.nn.modules import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

# Model Trainer
class ModelTrainer:

    def __init__(self, model, device, train_loader, valid_loader, lr=1e-3, epochs=10, model_name="", load_checkpoint=True, checkpoint_dir=None):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = BCEWithLogitsLoss()
        self.lr = lr
        self.epochs = epochs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.best_model_weights = model.state_dict()
        # self.chekpoint_dir = checkpoint_dir

        
        self.best_f1 = float("-inf")
        self.best_pr_auc = float("-inf")
        self.best_ROC_auc = float("-inf")
        self.best_val_loss = float("inf")

        self.start_epoch = 0

        # Handle checkpoint directory and run directory (mmdd-hhmm-{model})
        if checkpoint_dir and os.path.exists(checkpoint_dir) and any(file.endswith('.pth.tar') for file in os.listdir(checkpoint_dir)):
            self.run_dir = checkpoint_dir
            print(f'Using checkpoint directory {checkpoint_dir}')
        else:
            self.model_name = model_name if model_name else model.__class__.__name__
            current_time = datetime.datetime.now().strftime("%m%d-%H%M")
            self.run_dir = f'/home/juval.gutknecht/Mosquito_Detection/Inception-ResNet-v2/runs/{current_time}-{self.model_name}' if not checkpoint_dir else checkpoint_dir
            os.makedirs(self.run_dir, exist_ok=True)
            print(f'Created a new run directory: {self.run_dir}')

        # Add a new final Layer if needed
        final_layer_name = "last_linear" if hasattr(self.model, 'last_linear') else "fc"
        num_ftrs = getattr(self.model, final_layer_name).in_features
        setattr(self.model, final_layer_name, nn.Linear(num_ftrs, 1).to(self.device))
                
        # if hasattr(self.model, 'fc'):
        #     nr_filters = self.model.fc.in_features
        #     self.model.fc = nn.Linear(nr_filters, 1).to(self.device)
        # elif hasattr(self.model, 'last_linear'):  # Fallback for models that might use this naming
        #     nr_filters = self.model.last_linear.in_features
        #     self.model.last_linear = nn.Linear(nr_filters, 1).to(self.device)
        # else:
        #     raise AttributeError("Model does not have 'fc' or 'last_linear' layers")

        self.checkpoint_path = os.path.join(self.run_dir, 'checkpoint.pth.tar')
        if load_checkpoint and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()
        else:
            print('No checkpoint loaded, starting from scratch.')

                
        self.writer = SummaryWriter(log_dir=self.run_dir)
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_logger': None,  # Add loss logger if needed
            'best_f1': self.best_f1,
            'best_pr_auc': self.best_pr_auc,
            'best_ROC_auc': self.best_ROC_auc,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch+1}')

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_path):
            print(f'=> Loading checkpoint {self.checkpoint_path}')
            checkpoint = torch.load(self.checkpoint_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_f1 = checkpoint['best_f1']
            self.best_pr_auc = checkpoint['best_pr_auc']
            self.best_ROC_auc = checkpoint['best_ROC_auc']
            self.best_val_loss = checkpoint['best_val_loss']
            print(f'=> Loaded checkpoint (epoch {checkpoint["epoch"]})')
        else:
            print(f'=> No checkpoint found at {self.checkpoint_path}')
            self.start_epoch = 0

    def train_step(self, x, y):
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device).float()
        y_hat = self.model(x)
        y_hat = y_hat.squeeze(1)
        loss = self.loss_fn(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        probabilities = torch.sigmoid(y_hat).detach().cpu().numpy()
        predictions = (probabilities >= 0.5).astype(np.float32)
        return loss.item(), predictions, y.cpu().numpy()

    def val_step(self, x, y):
        self.model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for x_batch, y_batch in self.valid_loader:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.unsqueeze(1).float().to(self.device) # Add unsqueeze to make it 2D
                y_hat = self.model(x_batch)
                loss = self.loss_fn(y_hat, y_batch)
                val_loss += loss.item()
                best_val_loss = val_loss / len(self.valid_loader)

                probabilities = torch.sigmoid(y_hat).cpu().numpy()
                predictions = probabilities.round()
                # print(predictions)
                all_predictions.extend(predictions.flatten())
                all_scores.extend(probabilities.flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())

        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall, precision)
        balance_point = np.argmax(precision[:-1] + recall[:-1])
        optimal_threshold = thresholds[balance_point]
        roc_auc = roc_auc_score(all_labels, all_scores)
        f1 = f1_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)

        return best_val_loss, all_predictions, all_labels, optimal_threshold, pr_auc, roc_auc, f1, mcc
    
    def log_confusion_matrix(self, all_preds, all_labels, epoch):

        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm_normalized, cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix of {self.run_dir}')
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

        self.writer.add_figure(f'Confusion Matrix of {self.run_dir}', 
                               fig, global_step=epoch)
        plt.close(fig)

    def log_validation_samples(self, epoch):
        if epoch % 1 == 0:
            self.model.eval()
            images, labels = next(iter(self.valid_loader))
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs).cpu()
                predictions = (probabilities >= 0.5).int()

            positive_indices = labels == 1
            positive_images = images[positive_indices]
            positive_probabilities = probabilities[positive_indices]
            positive_predictions = predictions[positive_indices]

            num_images = min(8, positive_images.size(0))
            if num_images == 0:
                print('No positive samples in this batch')
                return
            
            cols = min(4, num_images)
            rows = 2
            img_width, img_height = positive_images.size(3), positive_images.size(2)
            collage = Image.new('L', (cols * img_width, rows * img_height)) # 'L' for grayscale

            for i in range(num_images):
                regular_image = positive_images[i, 0, :, :]
                background_subtracted_image = positive_images[i, 1, :, :]
                prediction = positive_predictions[i].item()
                confidence = positive_probabilities[i].item()
                true_label = labels[positive_indices][i].item()

                text = f'P: {prediction}\nC: {confidence:.2f}\nT: {true_label}'

                img_pil = TF.to_pil_image(regular_image.cpu())
                img_pil_background_subtracted = TF.to_pil_image(background_subtracted_image.cpu())

                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), text, fill = "white")

                # draw_bs = ImageDraw.Draw(img_pil_background_subtracted) # not necessary, because same image is used
                # draw_bs.text((10, 10), text, fill = "white")

                col_index = i % cols
                row_index = i // cols
                x_offset = col_index * img_width
                y_offset_regular = row_index * 2 * img_height
                y_offset_background_subtracted = y_offset_regular + img_height

                collage.paste(img_pil, (x_offset, y_offset_regular))
                collage.paste(img_pil_background_subtracted, (x_offset, y_offset_background_subtracted))
            
            collage_tensor = TF.to_tensor(collage)
            # self.writer.add_image(f'Positive Validation Sample', collage_tensor, global_step=epoch)
            


    def train(self):
        early_stopping_counter = 0

        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0.0
            total_positives = 0
            total_samples = 0
            all_train_preds = []
            all_train_labels = []
            
            for x_batch, y_batch, *_ in tqdm(self.train_loader, 
                                              desc=f'Epoch {epoch+1}/{self.epochs}'):
                num_positives_batch = (y_batch == 1).sum().item()
                num_total_batch = len(y_batch)
                total_positives += num_positives_batch
                total_samples += num_total_batch
                
                loss, predictions, labels = self.train_step(x_batch, y_batch)
                epoch_loss += loss

                all_train_preds.extend(predictions)
                all_train_labels.extend(labels)

            average_epoch_loss = epoch_loss / len(self.train_loader)
            train_f1 = f1_score(all_train_labels, all_train_preds)
            # Validation
            best_val_loss, all_predictions, all_labels, optimal_threshold, pr_auc, roc_auc, f1, mcc = self.val_step(x_batch, y_batch)
            
            print(f'Epoch {epoch+1}/{self.epochs} - Loss: {average_epoch_loss:.4f} - Validation Loss: {best_val_loss:.4f}\nF1: {f1:.4f} - PR AUC: {pr_auc:.4f} - ROC AUC: {roc_auc:.4f} - MCC: {mcc:.4f}')
            
            # Log metrics to tensorboard
            self.writer.add_scalar('Average training loss per Epoch', average_epoch_loss, epoch)
            self.writer.add_scalar('Average validation loss per Epoch', best_val_loss, epoch)
            self.writer.add_scalar('F1 Score Training', train_f1, epoch)
            self.writer.add_scalar('F1 Score Validation', f1, epoch)
            self.writer.add_scalar('PR AUC', pr_auc, epoch)
            self.writer.add_scalar('ROC AUC', roc_auc, epoch)
            self.writer.add_scalar('MCC', mcc, epoch)
            self.writer.add_scalar('Optimal Threshold', optimal_threshold, epoch)
            self.log_confusion_matrix(all_predictions, all_labels, epoch)
            # self.log_validation_samples(epoch)

            # Save the model if it has the best F1 score
            if f1 > self.best_f1 or \
                (f1 == self.best_f1 and pr_auc > self.best_pr_auc) or \
                (f1 == self.best_f1 and pr_auc == self.best_pr_auc and roc_auc > self.best_ROC_auc) or \
                (f1 == self.best_f1 and pr_auc == self.best_pr_auc and roc_auc == self.best_ROC_auc and best_val_loss < self.best_val_loss):
                self.best_f1 = f1
                self.best_pr_auc = pr_auc
                self.best_ROC_auc = roc_auc
                self.best_val_loss = best_val_loss

                # Save the model
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, 'best_model.pth'))
                self.save_checkpoint(epoch)
                print(f'Best model updated and saved with \nValidation F1: {f1:.4f}, PR AUC: {pr_auc:.4f}, \nROC AUC: {roc_auc:.4f}, Val Loss: {best_val_loss:.4f}')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1 # early stopping is not implemented


        self.writer.close()
        print('Training finished')


# Initialize the model
device = "cuda:6" if torch.cuda.is_available() else "cpu"
model = InceptionResNetV2(num_classes = 2)
# model = models.resnet101(num_classes = 2)
checkpoint_dir = ''
trainer = ModelTrainer(model, device, train_loader, valid_loader, lr=1e-3, epochs=2000, model_name="dataset-combined", checkpoint_dir=checkpoint_dir)

trainer.train()


# CUDA_VISIBLE_DEVICES=3 python training_v0.py
# unset CUDA_VISIBLE_DEVICES to reset or export CUDA_VISIBLE_DEVICES=  (empty string) to unset