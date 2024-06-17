import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, seed=42):
        super(Attention, self).__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.attention = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_len, input_dim)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        energy = torch.tanh(
            self.attention(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attention_scores = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(
            1)  # (batch_size, input_dim)
        return context_vector, attention_weights


class ECG_CNN1D_LSTM_att(nn.Module):
    def __init__(self, alpha=0.1, embedded_size=64, dropout=0.1, output_size=1, seed=42):
        super(ECG_CNN1D_LSTM_att, self).__init__()


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

        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(input_dim=256, hidden_dim=256, seed=seed)

        self.fc1 = nn.Linear(256, 256)  # Adjusted input size
        self.fc2 = nn.Linear(256, embedded_size)
        self.fc3 = nn.Linear(embedded_size, output_size)
        self.alpha = alpha

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout * 2)
        self.dropout_fc2 = nn.Dropout(dropout * 2)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 1250, 64)
        lstm_out, (hn, _) = self.lstm(x)
        context_vector, attention_weights = self.attention(lstm_out[:, -1, :],
                                                           lstm_out)  # Use last hidden state of LSTM
        x = self.dropout_fc1(F.leaky_relu(self.fc1(context_vector), negative_slope=self.alpha))
        x = self.dropout_fc2(F.leaky_relu(self.fc2(x), negative_slope=self.alpha))
        x = torch.sigmoid(self.fc3(x))
        return x



