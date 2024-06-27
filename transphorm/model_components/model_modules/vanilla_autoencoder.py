from torch import nn
import torch
import lightning as L
import numpy as np
import matplotlib.pyplot as plt


class CNNENcoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=100),
            nn.MaxPool1d(100),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=10),
            nn.Dropout1d(0.5),
        )


class Encoder(nn.Module):
    def __init__(self, seq_length, smallest_layer):
        super().__init__()
        self.seq_length = seq_length
        self.smallest_layer = smallest_layer
        self.linear_layers = nn.Sequential(
            nn.Linear(seq_length, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, smallest_layer),
        )

    def forward(self, x):
        return self.linear_layers(x)


class Decoder(nn.Module):
    def __init__(self, seq_length, smallest_layer):
        super().__init__()
        self.seq_length = seq_length
        self.smallest_layer = smallest_layer

        self.linear_layers = nn.Sequential(
            nn.Linear(smallest_layer, 512), nn.ReLU(), nn.Linear(512, seq_length)
        )

    def forward(self, x):
        return self.linear_layers(x)


class VanillaAutoEncoder(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        seq_length: int,
        smallest_layer: int,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()

        # store input and output size
        self.seq_length = seq_length
        self.smallest_layer = smallest_layer

        # inti encoder and decoder
        self.encoder = encoder(
            seq_length=self.seq_length, smallest_layer=self.smallest_layer
        )
        self.decoder = decoder(
            seq_length=self.seq_length, smallest_layer=self.smallest_layer
        )

        # set optimizer and loss
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()

        # save model hyperparams
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    # what metrics?
    def forward(self, inputs):
        x = self.encoder(inputs)
        recon_x = self.decoder(x)
        return recon_x

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def _common_step(self, batch, batch_idx):
        X = batch[0]
        encoded = self.encoder(X)
        x_hat = self.decoder(encoded)
        loss = self.loss_fn(x_hat, X)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("Validation loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def _plot_example_trace(self, X):
        exp = X.cpu().detach().numpy()
        plt.plot(exp[0])
        plt.show()

    # def on_train_epoch_end(self):
    # avg_loss = torch.mean(torch.Tensor(self.loss)).item()
    # self.loss = []
