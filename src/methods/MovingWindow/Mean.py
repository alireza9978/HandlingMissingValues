import numpy as np
import pandas as pd

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel

window_size = None


def fill_nan(temp_df: pd.DataFrame):
    import swifter
    _ = swifter.config
    temp_nan_index = temp_df.usage.isna()
    final_temp_nan_index = temp_df.index[temp_nan_index].to_numpy()
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    half_window_size = int(window_size / 2)

    def inner_window_filler(nan_row):
        row_index = nan_row["row_index"]
        usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
                                       temp_array[row_index - half_window_size:row_index]])
        return np.nanmean(usage_window).sum()

    temp_df["row_index"] = temp_df.index
    filled_nan = temp_df[temp_nan_index].swifter.progress_bar(False). \
        apply(inner_window_filler, axis=1).to_numpy().reshape(-1, 1)
    return pd.Series([filled_nan, final_temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset("0.01")
    window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]
    # window_sizes = [4, 6, 8, 10, 12, 24, 48]
    for window_size in window_sizes:
        filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        print("window size = ", window_size)
        print(evaluate_dataframe(filled_users, mean_square_error))
        print()
