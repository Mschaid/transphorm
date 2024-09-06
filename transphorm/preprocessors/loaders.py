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

    def __init__(self, path: str, down_sample: bool = False) -> None:
        """
        Initialize the AADataLoader with a file path.

        Args:
            path (str): The file path to the data.
        """
        self.path: str = path
        self.data: Optional[Tensor] = None
        self.x: Optional[List[Tensor]] = None
        self.labels: Optional[Tensor] = None
        self.down_sample = down_sample

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
        self.data = torch.load(self.path).detach().numpy()

    def prepare_data(self) -> None:
        """
        Prepare the loaded data for further processing.

        This method reshapes the feature data into a list of 2D tensors.
        """
        self._clean_data()
        if self.down_sample:
            self.x = self.data[:, 1:][:, ::10]
        else:
            self.x = self.data[:, 1:]

        self.x = [self.x[i].reshape(-1, 1) for i in range(self.x.shape[0])]
        self.labels = self.data[:, 0]
