import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_dataset


def get_name():
    return "moving_window_exponential_mean"


def get_params():
    return [4, 6, 8, 10, 12, 24, 48, 126]


def fill_nan(temp_df: pd.DataFrame, window_size):
    if window_size is None:
        return None

    import swifter
    _ = swifter.config
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
    temp_df = temp_df.reset_index(drop=True)
    half_window_size = int(window_size / 2)
    temp_array = np.pad(temp_array.squeeze(), half_window_size, mode="constant", constant_values=np.nan).reshape(-1, 1)
    bottom = np.power(np.full(window_size, 2, dtype=np.uint64), np.concatenate(
        [np.arange(half_window_size, 0, -1, dtype=np.uint64),
         np.arange(1, half_window_size + 1, dtype=np.uint64)]), dtype=np.uint64)
    weights = np.divide(np.ones(bottom.shape[0], dtype=np.uint64), bottom)

    def inner_window_filler(nan_row):
        row_index = nan_row["row_index"] + half_window_size
        usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
                                       temp_array[row_index - half_window_size:row_index]])

        not_nan_mask = ~np.isnan(usage_window)
        usage_window = usage_window[not_nan_mask]
        if usage_window.shape[0] > 0:
            weights_sum = np.sum(weights[not_nan_mask.squeeze()])
            return (usage_window.squeeze() * weights[not_nan_mask.squeeze()]).sum() / weights_sum
        return np.nan

    temp_df["row_index"] = temp_df.index
    temp_nan_index = temp_df.usage.isna()

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
