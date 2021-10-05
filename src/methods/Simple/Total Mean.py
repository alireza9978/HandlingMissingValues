import numpy as np

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_array: np.ndarray):
    temp_mean = np.nanmean(temp_array).sum()
    return np.nan_to_num(temp_array, nan=temp_mean)


if __name__ == '__main__':
    x, x_nan = get_dataset()
    x_filled_nan = fill_nan(x_nan)
    print(mean_square_error(x, x_filled_nan))