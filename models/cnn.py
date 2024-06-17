import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np



class ECG_CNN_1D(nn.Module):
    def __init__(self, alpha=0.1, embedded_size=64, output_size=1, dropout=0.1, seed=42):
        super(ECG_CNN_1D, self).__init__()

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
        self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)  # Adjusted input size
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

    def _get_flatten_size(self):
        # Forward pass a dummy tensor to get the output shape after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, 5000)  # Adjusted input shape
            x = F.leaky_relu(self.conv1(dummy_input))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            x = self.avgpool(x)
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=self.alpha))
        x = self.dropout_fc2(F.leaky_relu(self.fc2(x), negative_slope=self.alpha))
        x = torch.sigmoid(self.fc3(x))
        return x

