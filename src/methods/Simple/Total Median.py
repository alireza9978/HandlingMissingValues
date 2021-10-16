import numpy as np
import pandas as pd
from src.measurements.Measurements import mean_square_error, evaluate_dataframe
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_df: pd.DataFrame):
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    temp_mean = np.nanmedian(temp_array).sum()
    filled_nan = np.nan_to_num(temp_array, nan=temp_mean)
    temp_nan_index = np.where(np.isnan(temp_array))[0]
    return pd.Series([filled_nan[temp_nan_index], temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset()
    filled_users = x_nan.groupby("id").apply(fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))