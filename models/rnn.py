import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class ECG_RNN(nn.Module):
    def __init__(self, dropout=0.2, hidden_size=128, output_size=1, num_layers=2, seed=42):
        super(ECG_RNN, self).__init__()


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout_fc1 = nn.Dropout(dropout)
        # self.dropout_fc2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 5000, 12)
        _, (hn, _) = self.lstm(x)
        hn = hn[-2:].permute(1, 0, 2).contiguous().view(x.size(0), -1)  # Concatenate hidden states from both directions
        x = self.dropout_fc1(F.relu(self.fc1(hn)))
        x = torch.sigmoid(self.fc2(x))
        return x