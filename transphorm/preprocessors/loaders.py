from scipy import signal
import torch
from torch import Tensor
from typing import List, Optional

import numpy as np


class AADataLoader:
    """
    A data loader class for handling and preprocessing AA (Assumed to be Alcohol Addiction) data.

    This class loads data from a specified path, extracts features and labels,
    and prepares the data for further processing or model input.

    Attributes:
        path (str): The file path to the data.
        data (Optional[Tensor]): The raw data loaded from the file.
        x (Optional[List[Tensor]]): The processed feature data.
        labels (Optional[Tensor]): The labels extracted from the data.
    """

    def __init__(
        self,
        path: str,
        down_sample: bool = False,
        low_pass: bool = False,
        down_sample_factor: int = 100,
    ) -> None:
        """
        Initialize the AADataLoader with a file path.

        Args:
            path (str): The file path to the data.
        """
        self.path: str = path
        self.data: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.train: Optional[List[np.ndarray]] = None
        self.test: Optional[List[np.ndarray]] = None
        self.labels: Optional[np.ndarray] = None
        self.down_sample = down_sample
        self.down_sample_factor = down_sample_factor
        self.low_pass = low_pass

    def _clean_data(self) -> None:
        """
        Clean the data by removing NaNs and infs.
        """
        try:
            self.data = self.data[~np.isnan(self.data).any(axis=1)]
        except:
            pass
        try:
            self.data = self.data[~np.isinf(self.data).any(axis=1)]
        except:
            pass

    def load_data(self) -> None:
        """
        Load the data from the specified file path..
        """
        self.data = torch.load(self.path)

        if type(self.data) == torch.Tensor:
            self.data = self.data.detach().numpy()

    def _apply_low_pass(
        self, cutoff_frequency: float = 0.8, sampling_rate: float = 1000
    ) -> None:
        """
        Apply a low pass filter to the data.
        """
        nyquist_frequency = 0.5 * sampling_rate
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Design the Butterworth low-pass filter
        b, a = signal.butter(N=2, Wn=normalized_cutoff, btype="low", analog=False)

        # Apply the filter
        self.x = signal.filtfilt(b, a, self.x, axis=1)

    def _down_sample(self) -> None:
        """
        Down sample the data.
        """
        self.x = signal.decimate(self.x, self.down_sample_factor, axis=1)

    def _shape_for_arhmm(self, x) -> None:
        return [x[i].reshape(-1, 1) for i in range(x.shape[0])]

    def prepare_data(self) -> None:
        """
        Prepare the loaded data for further processing.

        This method reshapes the feature data into a list of 2D tensors.


        """

        self._clean_data()
        self.x = self.data[:, 1:]
        self.labels = self.data[:, 0]

        if self.low_pass:
            self._apply_low_pass()
        if self.down_sample:
            self._down_sample()

        len_30_min = self.x.shape[1]
        len_1_min = len_30_min // 30
        training_idx = int(len_1_min * 25)

        self.train = self._shape_for_arhmm(self.x[:, :training_idx])
        self.test = self._shape_for_arhmm(self.x[:, training_idx:])
        self.labels = self.data[:, 0]
