from torch import nn
import torch
import lightning as L


# define LSTM autoencoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm_layers = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    # forward pass
    def forward(self, x):
        # initialize hidden state
        h0 = torch.zeros(1, x.shape[0], self.lstm_layers.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.shape[0], self.lstm_layers.hidden_size).to(x.device)
        # forward pass
        out, _ = self.lstm_layers(x, (h0, c0))
        out = self.linear_layer(out[:, -1, :])
        return out


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm_layers = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.linear_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # forward pass
    def forward(self, x):
        # initialize hidden state
        h0 = torch.zeros(1, x.shape[0], self.lstm_layers.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.shape[0], self.lstm_layers.hidden_size).to(x.device)
        # forward pass
        out, _ = self.lstm_layers(x, (h0, c0))
        out = self.linear_layer(out[:, -1, :])
        return out


class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(LSTMAE, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, output_dim)

    # forward pass
    def forward(self, x):
        # encoder
        latent = self.encoder(x)
        # decoder
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# simple test of LSTM autoencoder
if __name__ == "__main__":
    # define input and output dimensions
    input_dim = 10
    output_dim = 10
    hidden_dim = 10
    latent_dim = 10
    # define model
    model = LSTMAE(input_dim, hidden_dim, latent_dim, output_dim)
    # define input
    x = torch.randn(10, input_dim)
    # forward pass
    reconstruction, latent = model(x)
    print(reconstruction.shape)
    print(latent.shape)
