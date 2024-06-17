#import

from torch.utils.data import DataLoader
import glob
import random
from utils import ECGDataset
from eval import record_metrics, plot_metrics, save_metrics
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

# import models
from models.cnn import ECG_CNN_1D
from models.rnn import ECG_RNN
from models.cnn_rnn import ECG_CNN1D_LSTM
from models.cnn_att_rnn import ECG_CNN1D_LSTM_att
from models.cnn_transformer import ECG_CNN1D_Transformer





def train(model, train_data_path, test_data_path, model_name='ECG_CNN_1D', batch_size=32, num_epochs=30, lr=0.0001, seed=42):

    # Ensure reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    # Create data loaders
    train_files = train_data_path
    test_files = test_data_path

    train_dataset = ECGDataset(train_files)
    test_dataset = ECGDataset(test_files)

    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)




    # Initialize device, model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = num_epochs

    # Initialize dictionaries to store metrics
    metrics = {
        'epoch': [],
        'train_loss': [], 'train_accuracy': [], 'train_precision': [],
        'train_recall': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_auc': []
    }


    save_folder = os.path.join(os.getcwd(), model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    best_val_auc = -1.0
    best_metrics = {}

    for epoch in range(num_epochs):
        print()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for signals, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            signals, labels = signals.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        record_metrics('train', train_loss, train_preds, train_labels, metrics, epoch, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for signals, labels in tqdm(test_loader, desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
                signals, labels = signals.to(device), labels.to(device).unsqueeze(1)
                outputs = model(signals)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        record_metrics('val', val_loss, val_preds, val_labels, metrics, epoch, len(test_loader))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {metrics['train_loss'][-1]:.4f}, "
              f"Train Accuracy: {metrics['train_accuracy'][-1]:.4f}, Train Precision: {metrics['train_precision'][-1]:.4f}, "
              f"Train Recall: {metrics['train_recall'][-1]:.4f}, Train F1: {metrics['train_f1'][-1]:.4f}, Train AUC: {metrics['train_auc'][-1]:.4f}, "
              f"Val Loss: {metrics['val_loss'][-1]:.4f}, Val Accuracy: {metrics['val_accuracy'][-1]:.4f}, "
              f"Val Precision: {metrics['val_precision'][-1]:.4f}, Val Recall: {metrics['val_recall'][-1]:.4f}, "
              f"Val F1: {metrics['val_f1'][-1]:.4f}, Val AUC: {metrics['val_auc'][-1]:.4f}")


        # Check if this model has the best validation AUC
        if metrics['val_auc'][-1] > best_val_auc:
            best_val_auc = metrics['val_auc'][-1]
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': metrics['train_loss'][-1],
                'train_accuracy': metrics['train_accuracy'][-1],
                'train_precision': metrics['train_precision'][-1],
                'train_recall': metrics['train_recall'][-1],
                'train_f1': metrics['train_f1'][-1],
                'train_auc': metrics['train_auc'][-1],
                'val_loss': metrics['val_loss'][-1],
                'val_accuracy': metrics['val_accuracy'][-1],
                'val_precision': metrics['val_precision'][-1],
                'val_recall': metrics['val_recall'][-1],
                'val_f1': metrics['val_f1'][-1],
                'val_auc': metrics['val_auc'][-1]
            }
            #torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))


    # Convert metrics to DataFrame
    metrics['epoch'] = list(range(1, num_epochs + 1))
    metrics_df = pd.DataFrame(metrics)



    # Save metrics and plots
    save_metrics(metrics_df, model_name, save_folder)

    # Plotting example
    plot_metrics(metrics_df, 'loss', model_name, save_folder)
    plot_metrics(metrics_df, 'accuracy', model_name, save_folder)
    plot_metrics(metrics_df, 'precision', model_name, save_folder)
    plot_metrics(metrics_df, 'recall', model_name, save_folder)
    plot_metrics(metrics_df, 'f1', model_name, save_folder)
    plot_metrics(metrics_df, 'auc', model_name, save_folder)



    # Print the best model's performance metrics
    print("\nBest Model Performance Metrics:")
    print(f"Epoch: {best_metrics['epoch']}")
    print(f"Train Loss: {best_metrics['train_loss']:.4f}, Train Accuracy: {best_metrics['train_accuracy']:.4f}, "
          f"Train Precision: {best_metrics['train_precision']:.4f}, Train Recall: {best_metrics['train_recall']:.4f}, "
          f"Train F1: {best_metrics['train_f1']:.4f}, Train AUC: {best_metrics['train_auc']:.4f}")
    print(f"Val Loss: {best_metrics['val_loss']:.4f}, Val Accuracy: {best_metrics['val_accuracy']:.4f}, "
          f"Val Precision: {best_metrics['val_precision']:.4f}, Val Recall: {best_metrics['val_recall']:.4f}, "
          f"Val F1: {best_metrics['val_f1']:.4f}, Val AUC: {best_metrics['val_auc']:.4f}")



def main():

    train_data_path = '/Volumes/research/ecg_echo/echo_data/train_abnormal_ecgs/*'
    test_data_path = '/Volumes/research/ecg_echo/echo_data/test_abnormal_ecgs/*'
    model = ECG_CNN1D_LSTM_att()
    train(model=model, model_name='ECG_CNNID_Attention_LSTM', train_data_path=train_data_path, test_data_path=test_data_path, batch_size=32, num_epochs=30, lr=0.0001, seed=42)


if __name__ == "__main__":
    main()