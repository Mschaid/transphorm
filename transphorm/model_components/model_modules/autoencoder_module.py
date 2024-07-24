import torch
from torch import nn
import lightning as L


class AutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, optimizer, learning_rate):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.train_loss = []
        self.val_loss = []

    def forward(self, inputs):
        x = self.encoder(inputs)
        x_recon = self.decoder(x)
        return x_recon

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _common_step(self, batch, batch_idx):
        X = batch[0]
        x_hat = self.forward(X)
        loss = self.loss_fn(x_hat, X)

        return loss

    def training_step(self, batch, batch_idx):

        loss = self._common_step(batch, batch_idx)
        self.train_loss.append(loss)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.val_loss.append(loss)
        return loss
