import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc_roc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, auc_roc

def record_metrics(phase, loss, preds, labels, metrics, epoch, loader_len):
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy, precision, recall, f1, auc = compute_metrics(labels, preds)
    metrics[f'{phase}_loss'].append(loss / loader_len)
    metrics[f'{phase}_accuracy'].append(accuracy)
    metrics[f'{phase}_precision'].append(precision)
    metrics[f'{phase}_recall'].append(recall)
    metrics[f'{phase}_f1'].append(f1)
    metrics[f'{phase}_auc'].append(auc)

def plot_metrics(metrics_df, metric_name, model_name, save_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['epoch'], metrics_df[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
    plt.plot(metrics_df['epoch'], metrics_df[f'val_{metric_name}'], label=f'Val {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} over Epochs - model: {model_name}')
    plt.legend()
    plt.savefig(os.path.join(save_folder, f'{metric_name}_plot.png'))
    plt.show()
    plt.close()

def save_metrics(metrics_df, model_name, save_folder):
    metrics_df.to_csv(os.path.join(save_folder, 'model_metrics.csv'), index=False)