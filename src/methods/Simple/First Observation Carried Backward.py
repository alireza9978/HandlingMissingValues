import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_array: np.ndarray):
    df = pd.DataFrame(temp_array)
    df.columns = ["data"]
    df = df.bfill()
    temp_nan_index = np.where(np.isnan(temp_array))[0]
    filled_nan = df["data"].to_numpy().reshape(-1, 1)
    return filled_nan[temp_nan_index], temp_nan_index


if __name__ == '__main__':
    x, x_nan = get_dataset()
    filled_x, nan_index = fill_nan(x_nan)
    print(mean_square_error(x[nan_index], filled_x))
