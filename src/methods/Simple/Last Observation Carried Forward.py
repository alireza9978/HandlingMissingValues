import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_array: np.ndarray):
    df = pd.DataFrame(temp_array)
    df.columns = ["data"]
    df = df.ffill()
    return df["data"].to_numpy().reshape(-1, 1)


if __name__ == '__main__':
    x, x_nan = get_dataset()
    x_filled_nan = fill_nan(x_nan)
    print(mean_square_error(x, x_filled_nan))
