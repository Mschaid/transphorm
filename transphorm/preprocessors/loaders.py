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
        butter_filter: bool = False,
        butter_cutoff: int = 2,
        butter_fs: int = 1070,
        butter_order: int = 6,
        smoothing: bool = False,
        smoothing_window_size: int = 535,
        weiner_filter: bool = False,
        weiner_window_size: int = 556,
        down_sample: bool = False,
        down_sample_factor: int = 25,
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
        self.butter_filter = butter_filter
        self.butter_cutoff = butter_cutoff
        self.butter_fs = butter_fs
        self.butter_order = butter_order
        self.smoothing = smoothing
        self.smoothing_window_size = smoothing_window_size
        self.down_sample = down_sample
        self.down_sample_factor = down_sample_factor
        self.weiner_filter = weiner_filter
        self.weiner_window_size = weiner_window_size

    def _clean_data(self) -> None:
        """
        Clean the data by removing NaNs and infs, and clipping extreme values.
        """
        # Remove rows with NaN or inf values
        self.data = self.data[~np.isnan(self.data).any(axis=1)]
        self.data = self.data[~np.isinf(self.data).any(axis=1)]

        # Clip extreme values (optional, adjust as needed)
        # lower_bound = np.percentile(self.data, 1)
        # upper_bound = np.percentile(self.data, 99)
        self.data = self.data[~(self.data < -99.0).any(axis=1)]

        print(f"clean data call: min={np.min(self.data)}, max={np.max(self.data)}")

    def load_data(self) -> None:
        """
        Load the data from the specified file path..
        """
        self.data = torch.load(self.path)

        if type(self.data) == torch.Tensor:
            self.data = self.data.detach().numpy()
        # self.data = np.delete(self.data, [74, 75, 76, 77], axis=0)
        print(f"load data call: {np.min(self.data)}")

    def _butter_lowpass_filter(
        self, data, cutoff: int = 1.5, fs: int = 1070, order: int = 8
    ):
        """
        Applies a low-pass Butterworth filter to the input data.

        Parameters:
        data (np.array): The time series data to filter. Shape should be (n_samples,) or (n_trials, n_samples)
        cutoff (float): The cutoff frequency of the filter in Hz
        fs (float): The sampling frequency of the data in Hz
        order (int): The order of the filter (default is 6)

        Returns:
        np.array: The filtered data, same shape as input
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)

        if data.ndim == 1:
            filtered_data = signal.filtfilt(b, a, data)
        else:
            filtered_data = np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x), axis=-1, arr=data
            )
        print(f"butter call: {np.min(filtered_data)}")
        return filtered_data

    def _weiner_filter(self, noise_power: Optional[float] = None) -> None:

        self.x = np.array(
            [
                signal.wiener(row, mysize=self.weiner_window_size, noise=noise_power)
                for row in self.x
            ]
        )
        print(f"weiner call: {np.min(self.x)}")

    def __apply_smoothing(self, window_size: int) -> None:
        self.x = np.array(
            [
                signal.convolve(
                    row,
                    np.ones(self.smoothing_window_size) / self.smoothing_window_size,
                    mode="valid",
                )
                for row in self.x
            ]
        )
        print(f"smoothing call: {np.min(self.x)}")

    def _apply_smoothing(self) -> None:
        self.__apply_smoothing(self.smoothing_window_size)
        print(f"smoothing call: {np.min(self.x)}")
        # self.__apply_smoothing(self.smoothing_window_size // 2)
        # self.__apply_smoothing(self.smoothing_window_size // 4)

    def _down_sample(self) -> None:
        """
        Down sample the data.
        """
        self.x = signal.decimate(self.x, self.down_sample_factor, axis=1)
        print(f"down sample call: {np.min(self.x)}")

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

        if self.butter_filter:
            self.x = self._butter_lowpass_filter(
                self.x, self.butter_cutoff, self.butter_fs, self.butter_order
            )

        if self.smoothing:
            self._apply_smoothing()
        if self.weiner_filter:
            self._weiner_filter()
        if self.down_sample:
            self._down_sample()

        len_30_min = self.x.shape[1]
        len_1_min = len_30_min // 30
        training_idx = int(len_1_min * 25)

        self.train = self._shape_for_arhmm(self.x[:, :training_idx])
        self.test = self._shape_for_arhmm(self.x[:, training_idx:])
        self.labels = self.data[:, 0]
