import numpy as np

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_array: np.ndarray):
    temp_mean = np.nanmedian(temp_array).sum()
    filled_nan = np.nan_to_num(temp_array, nan=temp_mean)
    temp_nan_index = np.where(np.isnan(temp_array))[0]
    return filled_nan[temp_nan_index], temp_nan_index


if __name__ == '__main__':
    x, x_nan = get_dataset()
    filled_x, nan_index = fill_nan(x_nan)
    print(mean_square_error(x[nan_index], filled_x))
