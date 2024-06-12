def apply_smoothing_to_script():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Function to calculate the exponential moving average
    def exponential_moving_average(values, alpha=0.1):
        smoothed_values = []
        for i, value in enumerate(values):
            if i == 0:
                smoothed_values.append(value)
            else:
                previous_ema = smoothed_values[-1]
                new_ema = alpha * value + (1 - alpha) * previous_ema
                smoothed_values.append(new_ema)
        return smoothed_values

    # Load the CSV file to understand its structure
    file_path = '/home/juval.gutknecht/Mosquito_Detection/Git/YOLO/runs/detect/train5/results.csv'
    data = pd.read_csv(file_path)

    # Setting up the plot for loss metrics over epochs
    plt.figure(figsize=(14, 10))

    # Correcting the column names by including leading spaces
    epoch_col = '                  epoch'
    train_box_loss_col = '         train/box_loss'
    val_box_loss_col = '           val/box_loss'
    train_cls_loss_col = '         train/cls_loss'
    val_cls_loss_col = '           val/cls_loss'
    train_dfl_loss_col = '         train/dfl_loss'
    val_dfl_loss_col = '           val/dfl_loss'

    # Plotting training and validation losses with smoothing
    plt.subplot(3, 1, 1)
    plt.plot(data[epoch_col], exponential_moving_average(data[train_box_loss_col]), label='Train Box Loss')
    plt.plot(data[epoch_col], exponential_moving_average(data[val_box_loss_col]), label='Validation Box Loss', linestyle='--')
    plt.title('Box Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(data[epoch_col], exponential_moving_average(data[train_cls_loss_col]), label='Train Class Loss')
    plt.plot(data[epoch_col], exponential_moving_average(data[val_cls_loss_col]), label='Validation Class Loss', linestyle='--')
    plt.title('Class Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(data[epoch_col], exponential_moving_average(data[train_dfl_loss_col]), label='Train DFL Loss')
    plt.plot(data[epoch_col], exponential_moving_average(data[val_dfl_loss_col]), label='Validation DFL Loss', linestyle='--')
    plt.title('DFL Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('/home/juval.gutknecht/Mosquito_Detection/Git/YOLO/runs/detect/train5/', 'losses.png'))

    # Setting up the plot for accuracy metrics over epochs
    plt.figure(figsize=(14, 7))

    # Plotting precision and other metrics with smoothing
    plt.subplot(2, 2, 1)
    plt.plot(data[epoch_col], exponential_moving_average(data['   metrics/precision(B)']), label='Precision', color='blue')
    plt.title('Precision Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(data[epoch_col], exponential_moving_average(data['      metrics/recall(B)']), label='Recall', color='green')
    plt.title('Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(data[epoch_col], exponential_moving_average(data['       metrics/mAP50(B)']), label='mAP50', color='red')
    plt.title('mAP50 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(data[epoch_col], exponential_moving_average(data['    metrics/mAP50-95(B)']), label='mAP50-95', color='purple')
    plt.title('mAP50-95 Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50-95')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('/home/juval.gutknecht/Mosquito_Detection/Git/YOLO/runs/detect/train5/', 'metrics.png'))

# This is the function that wraps the entire updated script
apply_smoothing_to_script()
