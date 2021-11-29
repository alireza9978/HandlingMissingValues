import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_dataset


def get_name():
    return "moving_window_mean"


def get_params():
    return [4, 6, 8, 10, 12, 24, 48, 168, 720]


def fill_nan(temp_df: pd.DataFrame, window_size):
    import swifter
    _ = swifter.config

    final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
    temp_df = temp_df.reset_index(drop=True)
    temp_nan_index = temp_df.usage.isna()
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    half_window_size = int(window_size / 2)

    def inner_window_filler(nan_row):
        row_index = nan_row["row_index"]
        usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
                                       temp_array[row_index - half_window_size:row_index]])
        usage_window = usage_window[~np.isnan(usage_window)]
        if usage_window.shape[0] > 0:
            return np.nanmean(usage_window).sum()
        return np.nan

    temp_df["row_index"] = temp_df.index
    filled_nan = temp_df[temp_nan_index].apply(inner_window_filler, axis=1).to_numpy().reshape(-1, 1)

    temp_mean = np.nanmean(temp_array).sum()
    filled_nan = np.nan_to_num(filled_nan, nan=temp_mean)
    return pd.DataFrame({"predicted_usage": filled_nan.squeeze()},
                        index=final_temp_nan_index.squeeze())


if __name__ == '__main__':
    from src.utils.Methods import fill_nan as fn

    nan_percent = "0.01"
    x, x_nan = get_dataset(nan_percent)
    for temp_window_size in get_params():
        filled_users = fn(x, x_nan, fill_nan, temp_window_size)
        error, error_df = evaluate_dataframe_two(filled_users, mean_square_error)
        print(error)
        print(error_df)
