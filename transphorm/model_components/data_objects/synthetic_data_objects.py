from typing import List

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, TensorDataset


def z_score(dataset: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to the array.

    Args:
        dataset (np.ndarray): The input dataset.

    Returns:
        np.ndarray: The normalized dataset.
    """
    mean = np.mean(dataset, axis=0, keepdims=True)
    std = np.std(dataset, axis=0, keepdims=True)
    return (dataset - mean) / std


def generate_sequence() -> np.ndarray:
    """
    Generate a synthetic time series sequence. of length 5000
    injects random sine, square and saw tooth waves into the sequence and applies gaussian noise

    Returns:
        np.ndarray: The generated time series sequence.
    """
    sequence_length = 5000
    subsequence_length = np.random.randint(
        100, 200
    )  # Randomly generate the subsequence length
    num_subsequences = np.random.randint(
        20, 60
    )  # Randomly generate the number of subsequences

    # Generate the sine wave subsequence
    sine_wave = np.sin(
        np.linspace(0, np.random.randint(1, 10) * np.pi, subsequence_length)
    )

    # Generate the square wave subsequence
    square_wave = np.sign(
        np.sin(np.linspace(0, np.random.randint(1, 10) * np.pi, subsequence_length))
    )

    # Generate the saw tooth wave subsequence
    saw_tooth_wave = np.cumsum(np.random.uniform(-1, 1, subsequence_length))

    # Initialize the main sequence
    main_sequence = np.zeros(sequence_length)

    # Randomly insert the subsequences into the main sequence
    for _ in range(num_subsequences):
        subsequence_type = np.random.choice(
            ["sine", "square", "saw_tooth"]
        )  # Randomly choose the subsequence type
        if subsequence_type == "sine":
            subsequence = sine_wave
        elif subsequence_type == "square":
            subsequence = square_wave
        else:
            subsequence = saw_tooth_wave

        start_index = np.random.randint(
            0, sequence_length - subsequence_length
        )  # Randomly choose the start index
        main_sequence[start_index : start_index + subsequence_length] += subsequence

    # Add Gaussian noise to the main sequence
    main_sequence += np.random.normal(0, 0.01, sequence_length)
    return z_score(main_sequence.astype(np.float32))


def generate_dataset(numb_samples: int = 100_000) -> np.ndarray:
    """
    Generate a synthetic dataset with multiple time series sequences.

    Args:
        numb_samples (int, optional): The number of time series sequences to generate. Defaults to 10000.

    Returns:
        np.ndarray: The generated dataset.
    """
    dataset = np.array([generate_sequence() for _ in range(numb_samples)])
    return dataset


class SyntheticTSDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_loss = []

    def prepare_data(self):
        self.np_data = torch.from_numpy(generate_dataset())
        self.data = TensorDataset(self.np_data)

    def setup(self, stage="train"):
        self.train, self.val, self.test = torch.utils.data.dataset.random_split(
            self.data, [0.7, 0.2, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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

    def _plot_first_10(self):
        example_data = self.train.dataset.tensors[0][0:10, :].detach().numpy()
        fig, ax = plt.subplots(nrows=10, figsize=(16, 16), sharex=True, sharey=True)
        for i in range(10):
            ax[i].plot(
                example_data[i],
                color=np.random.rand(
                    3,
                ),
                lw=0.75,
            )
            sns.despine(left=True, right=True, top=True, bottom=True)
        plt.tight_layout()
