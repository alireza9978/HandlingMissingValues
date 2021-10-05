import numpy as np


def mean_square_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff2 = np.square(diff)
    diff2_mean = np.mean(diff2)
    return diff2_mean
