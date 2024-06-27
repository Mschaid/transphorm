from torch.utils.data.dataset import random_split
import torch
import lightning as L
from torch.utils.data import Dataset, TensorDataset, DataLoader
from pathlib import Path
import numpy as np


class AADataModule(L.LightningDataModule):
    def __init__(self, main_path: Path, batch_size: int, num_workers: int = 1):
        super().__init__()
        self.main_path = main_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = None

    def store_input_size(self, value):
        self.input_size = value

    def prepare_data(self, down_sample_factor=1):
        data = torch.load(self.main_path)

        data = data[~torch.isnan(data[:, 0])]

        features = data[:, 1:][:, ::down_sample_factor]
        labels = data[:, 0]

        input_size = features[0].shape[0]
        self.store_input_size(input_size)

        features = features.view(-1, 1, self.input_size)
        labels = labels.view(-1, 1)
        self.data = TensorDataset(
            features,
            labels,
        )

    def setup(self, stage):
        torch.manual_seed(42)
        self.train, self.val, self.test = random_split(self.data, [0.7, 0.15, 0.15])

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @property
    def tensor_shape(self):
        seq = self.train.dataset.tensors
        return seq[0].shape
