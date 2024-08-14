from typing import Any, Dict, Literal
import torch
from torch import nn
import torchmetrics as tm
import lightning as L


class TrialClassifer(L.LightningModule):
    def __init__(self, model, optimizer, learning_rate):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=['"classifier'])

    @property
    def metrics(self) -> Dict[str, tm.Metric]:

        accuracy = tm.classification.Accuracy(task="binary", num_classes=2)
        # confusion_matrix = tm.classification.ConfusionMatrix(task="binary")
        auroc = tm.classification.AUROC(task="binary")
        f1_score = tm.classification.F1Score(task="binary", num_classes=2)

        metrics = {
            "accuracy": accuracy,
            # "confusion_matrix": confusion_matrix,
            "auroc": auroc,
            "f1_score": f1_score,
        }
        return metrics

    def classify(self, logits):
        softmax = nn.Softmax(dim=0)
        y_prob = softmax(logits)
        y_hat = (y_prob > 0.5).float()
        return y_hat

    def compute_metrics(
        self,
        y: torch.Tensor,
        logits: torch.Tensor,
        group_log: Literal["train", "val", "test"],
    ) -> None:

        on_step = False
        on_epoch = True
        logger = True

        # compute logits to probabilities
        y_hat = self.classify(logits)

        device = y_hat.device
        for k, v in self.metrics.items():
            v.to(device=device)
            self.log(
                f"{group_log}_{k}",
                v(y, y_hat),
                on_step=on_step,
                on_epoch=on_epoch,
                logger=logger,
            )

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _common_step(self, batch, batch_idx, group_log):
        X, y = batch
        logits = self(X).view(-1)
        self.compute_metrics(y, logits, group_log)
        loss = self.loss_fn(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, group_log="train")

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, group_log="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X).view(-1)
        y_pred = self.classify(logits)
        return y_pred

    def on_after_backward(self):
        # Check for vanishing or exploding gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {name}")
