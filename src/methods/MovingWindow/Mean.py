import numpy as np
import pandas as pd

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel


def get_name():
    return "Moving_Window_Mean"


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
    return pd.Series([filled_nan, final_temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset("0.5")
    window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]
    # window_sizes = [4, 6, 8, 10, 12, 24, 48]
    for temp_window_size in window_sizes:
        filled_users = apply_parallel(x_nan.groupby("id"), fill_nan, temp_window_size)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        print("window size = ", temp_window_size)
        print(evaluate_dataframe(filled_users, mean_square_error))
        print()
