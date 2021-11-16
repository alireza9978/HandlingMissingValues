import numpy as np
import pandas as pd
from scipy import signal

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel

stds = [1, 2, 3, 4, 5, 6, 8, 12, 18]
window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]


def fill_nan(temp_df: pd.DataFrame, window_size):
    import swifter
    _ = swifter.config
    std = None
    for window_index in range(len(window_sizes)):
        if window_sizes[window_index] == window_size:
            std = stds[window_index]
    final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
    temp_df = temp_df.reset_index(drop=True)
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    half_window_size = int(window_size / 2)
    temp_array = np.pad(temp_array.squeeze(), half_window_size,
                        mode="constant", constant_values=np.nan).reshape(-1, 1)
    weights = signal.gaussian(window_size, std=std)

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
    return pd.Series([filled_nan, final_temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset("0.5")
    for i in range(len(window_sizes)):
        temp_window_size = window_sizes[i]
        filled_users = apply_parallel(x_nan.groupby("id"), fill_nan, temp_window_size)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        print("window size = ", temp_window_size)
        print(evaluate_dataframe(filled_users, mean_square_error))
        print()
