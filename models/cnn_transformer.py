import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ECG_CNN1D_Transformer(nn.Module):
    def __init__(self, alpha=0.1, embedded_size=64, num_layers=2, d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, output_size=1, seed=42):
        super(ECG_CNN1D_Transformer, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.conv1 = nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=7)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7)
        self.bn5 = nn.BatchNorm1d(64)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc1 = nn.Linear(d_model, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, embedded_size)
        self.fc3 = nn.Linear(embedded_size, output_size)
        self.alpha = alpha

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout*2)
        self.dropout_fc2 = nn.Dropout(dropout*2)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)  # Change shape to (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling on sequence dim
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=self.alpha))
        x = self.dropout_fc2(F.leaky_relu(self.fc2(x), negative_slope=self.alpha))
        x = torch.sigmoid(self.fc3(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
model = ECG_CNN1D_Transformer().to(device)
