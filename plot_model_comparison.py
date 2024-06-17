import os
import pandas as pd
import matplotlib.pyplot as plt

def read_metrics(folder_name):
    metrics_file = os.path.join(folder_name, 'model_metrics.csv')
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    else:
        print(f"No metrics found in {folder_name}")
        return None

def plot_val_comparison(metrics_dfs, metric_name, save_folder):
    plt.figure(figsize=(10, 5))
    for model_name, df in metrics_dfs.items():
        plt.plot(df['epoch'], df[f'val_{metric_name}'], label=f'{model_name} Val {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} - Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, f'val_{metric_name}_comparison_plot.png'))
    plt.show()

# Specify the folder names of your models
model_folders = ['ECG_CNN_1D', 'ECG_CNN1D_LSTM', 'ECG_CNN1D_LSTM_att', 'ECG_CNN1D_Transformer']

metrics_dfs = {}
for folder in model_folders:
    metrics_dfs[folder] = read_metrics(folder)

# Remove None values from the dictionary
metrics_dfs = {k: v for k, v in metrics_dfs.items() if v is not None}

# Specify the save folder for comparison plots
comparison_save_folder = os.path.join(os.getcwd(), 'comparison_plots')
if not os.path.exists(comparison_save_folder):
    os.makedirs(comparison_save_folder)

# Plot and save comparison metrics for validation only
for metric in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
    plot_val_comparison(metrics_dfs, metric, comparison_save_folder)





