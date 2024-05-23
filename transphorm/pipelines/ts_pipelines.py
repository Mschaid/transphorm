import numpy as np


def reshape_for_multivariable(arr: np.array) -> np.array:
    """
    Reshape a 2D array to a 3D array with one variable
    This function is used to enable all arrays to be used in multivariate inputs.

    Parameters
    ----------
    arr : np.array
        The 2D array to be reshaped.

    Returns
    -------
    np.array
        The reshaped 3D array.
    """
    return arr.reshape(*(s for s in arr.shape), 1)


def downsample(arr: np.array, keep_every_idx: int = 200) -> np.array:
    """
    Downsample a 3D array by keeping every index at a specified interval.

    This function is used for high fidelity timeseries.

    Parameters
    ----------
    arr : np.array
        The 3D array to be downsampled.
    keep_every_idx : int, optional
        The interval at which to keep indices, by default 200.

    Returns
    -------
    np.array
        The downsampled array.
    """
    return arr[:, ::200, :]


def process_arrays_for_ts(
    arr: np.array, indices_as_rows: bool = False, downsample_rate: int = 200
) -> np.array:
    """
    Preprocess X by reshaping and downsampling.

    This function reshapes a 2D array X into a 3D array with shape (s[0], s[1], 1),
    then downsamples it by keeping every index at the downsampled rate.

    Parameters
    ----------
    arr : np.array
        The 2D array to be processed.
    indices_as_rows : bool, optional
        If True, transpose the array before processing, by default False.
    downsample_rate : int, optional
        The rate at which to downsample the array, by default 200.

    Returns
    -------
    np.array
        The processed array.
    """
    if not indices_as_rows:
        arr = arr.T

    reshape_array = reshape_for_multivariable(arr)
    downsampled_arr = downsample(reshape_array, keep_every_idx=downsample_rate)
    return downsampled_arr
