from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import torch
import lightning as L

import numpy as np


class SyntheticFPDataModule(L.LightningDataModule):

    def __init__(
        self, batch_size: int = 32, num_workers: int = 1, sample_size: int = 1000
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_size = sample_size

    def create_sinewave(self, freq):
        """
        function to create a sine wave from a given frequency
        :param freq: frequency of the sine wave
        """
        # Define the number of points for the sine wave
        num_points = 1000

        random_multiplier = np.random.uniform(1, 10)
        # Create a sine wave
        x = np.linspace(0, random_multiplier * freq * np.pi, num_points)
        sine_wave = np.sin(x)

        # Create a vector of zeros with the same length as the sine wave
        zero_vector = np.zeros(num_points)

        # Find the indices where the sine wave crosses from negative to 0
        crossing_indices = np.where((np.diff(np.sign(sine_wave))) > 0)[0]

        # Set the corresponding values in the zero vector to 1 at the crossing points
        zero_vector[crossing_indices] = 1

        # Combine the sine wave and zero vector into a 2D matrix
        matrix = np.vstack((sine_wave, zero_vector))
        return matrix

    def create_square_wave(self, freq):
        """
        function to create a sine wave from a given frequency
        :param freq: frequency of the sine wave
        """
        num_points = 1000
        random_multiplier = np.random.uniform(1, 10)
        x = np.linspace(0, freq * random_multiplier * np.pi, num_points)
        square_wave = np.sign(np.sin(x * freq))
        zero_vector = np.zeros(num_points)
        crossing_indices = np.where((np.diff(np.sign(square_wave))) > 0)[0]
        zero_vector[crossing_indices] = 1
        matrix = np.vstack((square_wave, zero_vector))
        return matrix

    def create_dataset(self):
        """function to create synthetic dataset composed of sine and square waves, with labels"""

        squares = []
        sines = []

        for i in range(int((self.sample_size / 2))):
            random_freq = np.random.uniform(0, 1)
            sines.append(self.create_sinewave(random_freq))
            squares.append(self.create_square_wave(random_freq))
        X = np.vstack([sines, squares])

        sine_lables = np.zeros(int((self.sample_size / 2)))
        square_lables = np.ones(int((self.sample_size / 2)))
        y = np.hstack((sine_lables, square_lables))
        return X, y

    def prepare_data(self):
        X, y = self.create_dataset()
        X, y = torch.Tensor(X), torch.Tensor(y)
        self.data = TensorDataset(X, y)

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
