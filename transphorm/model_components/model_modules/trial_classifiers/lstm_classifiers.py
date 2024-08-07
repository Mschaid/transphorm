import torch
from torch import nn


class LSTMClassifer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout1d(0.2)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # init hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0).to(x.device)

        x_1, (h0, c0) = self.lstm(x, (h0, c0))
        batch_norm = self.batch_norm(h0[-1])
        dropout = self.dropout(batch_norm)
        out = self.fc(dropout)

        return out
