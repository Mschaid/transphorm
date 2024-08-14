from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import torch
import lightning as L

import numpy as np


class SyntheticFPDataModule(L.LightningDataModule):

    def __init__(self, data_path, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        X = torch.load(self.data_path)
        X = X.type(torch.float32)
        self.data = TensorDataset(X)

    def setup(self, stage):
        torch.manual_seed(42)
        self.train, self.val, self.test = random_split(self.data, [0.7, 0.2, 0.1])

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )
        return data_loader

    def test_dataloader(self):
        data_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return data_loader

    @property
    def tensor_shape(self):
        seq = self.train.dataset.tensors
        return seq[0].shape
