from abc import ABC, abstractmethod
from typing import Literal, Protocol

import torch
from torch import nn
import torchmetrics
import torch.functional as F
import lightning as L
import logging

import torchmetrics.classification

# set sup logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DeepLinearLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.linear_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_layers(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=100),
            nn.MaxPool1d(100),
            nn.Tanh(),
            nn.Conv1d(8, 8, kernel_size=10),
            nn.Dropout1d(0.5),
            nn.MaxPool1d(100),
            nn.Tanh(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(150, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # loss

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 150)
        x = self.linear_layers(x)
        return x


class DeepBinaryClassifier(L.LightningModule):
    def __init__(self, classifier, optimizer):
        super().__init__()
        self.save_hyperparameters()
        self.model = classifier
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer

        # metrics
        self.accuracy = torchmetrics.classification.Accuracy(
            task="binary", num_classes=2
        )

        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(
            task="binary"
        )

        self.auroc = torchmetrics.classification.AUROC(task="binary")

        self.f1_score = torchmetrics.classification.F1Score(
            task="binary", num_classes=2
        )

    def forward(self, inputs):
        return self.model(inputs)

    def classify(self, logits):
        sigmoid_output = self.model.output_activation(logits)
        threshold = 0.5
        classification = (sigmoid_output >= threshold).float()
        return classification

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y, logits = y.squeeze(), logits.squeeze()

        loss = self.loss_fn(logits, y)
        predictions = self.classify(logits)
        return loss, y, logits, predictions

    def _log_metrics(self, process: Literal["train", "val", "test"], y, predictions):

        self.log(
            f"{process}_accuracy",
            self.accuracy(y, predictions),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{process}_f1_score",
            self.f1_score(y, predictions),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            f"{process}_auroc",
            self.auroc(y, predictions),
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        # self.logger.experiment.log_confusion_matrix(
        #     f"{process}_confusion_matrix",
        #     self.confusion_matrix(y, predictions),
        #     title=f"{process}_confusion_matrix",
        # )

    def training_step(self, batch, batch_idx):
        loss, y, logits, predictions = self._common_step(batch, batch_idx)

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self._log_metrics("train", y, predictions)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, logits, predictions = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        self._log_metrics("val", y, predictions)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=0.0001, weight_decay=0.001)
        return optimizer
