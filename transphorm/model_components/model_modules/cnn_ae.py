from torch import nn
import lightning as L
import torch


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(start_dim=-2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        c = self.cnn_layers(x)
        e = self.linear_layers(c)
        return e


class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3136),
            nn.ReLU(),
            nn.Unflatten(dim=-1, unflattened_size=(16, -1)),
        )

        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=10, stride=5),
        )

    def forward(self, x):
        ld = self.linear_decoder(x)
        return self.cnn_decoder(ld)


class AutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder, optimizer, learning_rate):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder()
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
        self.log("Training loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("Validaiton loss", loss, on_step=False, on_epoch=True, logger=True)
        self.val_loss.append(loss)
        return loss

    # def on_train_epoch_end(self) -> None:
    #     avg_loss = torch.Tensor(self.train_loss)
    #     print(f"training loss : {avg_loss.mean()}")
    #     self.train_loss = []

    # def on_validation_epoch_end(self) -> None:
    #     avg_loss = torch.Tensor(self.val_loss)
    #     print(f"val loss : {avg_loss.mean()}")
    #     self.val_loss = []
