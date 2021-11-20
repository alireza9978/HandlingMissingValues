import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error, evaluate_dataframe, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_dataset


def fill_nan(temp_df: pd.DataFrame):
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1).copy()
    final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
    temp_nan_index = np.where(np.isnan(temp_array))[0]
    temp_df['usage'] = temp_df['usage'].bfill().ffill()
    filled_nan = temp_df["usage"].to_numpy()
    return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index]}, index=final_filled_nan_index.squeeze())


if __name__ == '__main__':
    from src.utils.Methods import fill_nan as fn

    nan_percent = "0.01"
    x, x_nan = get_dataset(nan_percent)
    filled_users = fn(x, x_nan, fill_nan)
    error, error_df = evaluate_dataframe_two(filled_users, mean_square_error)
    print(error)
    print(error_df)


def get_name():
    return "first_observation_carried_backward"
