
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_learning_curves(folder_path):
    # Initialize the plot
    folder_name = os.path.basename(folder_path.strip('/'))
    plt.figure(figsize=(10, 6))
    
    # List all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Load the CSV file
            data = pd.read_csv(file_path)
            # Plot the data
            plt.plot(data['Step'], data['Value'], label=f'{filename[:-4]}')  # Strip '.csv' from filename for label
    
    # Label the plot
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Curves {folder_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(curves_folder, f"{folder_name}-F1-curves.png"), bbox_inches='tight')
    plt.show()


def exponential_moving_average(values, alpha=0.1):
    """
    Calculate the exponential moving average of a list of values.
    
    :param values: List of numerical values (floats or integers).
    :param alpha: Smoothing factor.
    :return: List of smoothed values.
    """
    smoothed_values = []
    for i, value in enumerate(values):
        if i == 0:
            smoothed_values.append(value)
        else:
            # EMA formula: S_t = alpha * Y_t + (1 - alpha) * S_t-1
            previous_ema = smoothed_values[-1]
            new_ema = alpha * value + (1 - alpha) * previous_ema
            smoothed_values.append(new_ema)
    return smoothed_values

def plot_and_save_smoothed_learning_curves(folder_path):
    # Extract the last part of the folder path to use as a base for title and filename
    folder_name = os.path.basename(folder_path.strip('/'))
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    
    # List all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Load the CSV file
            data = pd.read_csv(file_path)
            # Apply smoothing
            smoothed_values = exponential_moving_average(data['Value'])
            # Plot the smoothed data
            plt.plot(data['Step'], smoothed_values, label=f'{filename[:-4]}')  # Strip '.csv' from filename for label
    
    # Label the plot
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{folder_name} F1 Curves')
    plt.legend()
    plt.grid(True)
    
    # Save the figure in the same folder with a dynamic name
    plt.savefig(os.path.join(folder_path, f"{folder_name}-F1-curves.png"), bbox_inches='tight')
    # plt.show()




# To use the function, uncomment the line below and replace 'your_folder_path' with the path to your folder
curves_folder = '/Users/Juval/Downloads/F1-score-training-curves/Training'

# plot_learning_curves(curves_folder)

plot_and_save_smoothed_learning_curves(curves_folder)