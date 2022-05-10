import numpy as np
import pandas as pd

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.smart_star.load_dataset import get_dataset


def fill_nan(temp_df: pd.DataFrame, window_size):
    import swifter
    _ = swifter.config
    temp_nan_index = temp_df.usage.isna()
    final_temp_nan_index = temp_df.index[temp_nan_index].to_numpy()
    temp_df = temp_df.reset_index(drop=True)
    temp_nan_index = temp_df.usage.isna()
    temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
    half_window_size = int(window_size / 2)
    temp_array = np.pad(temp_array.squeeze(), half_window_size,
                        mode="constant", constant_values=np.nan).reshape(-1, 1)

    temp_df["hour"] = temp_df.date.dt.hour
    temp_usage_mean = temp_df[["hour", "usage"]].groupby("hour").mean()

    def inner_window_filler(nan_row):
        # row_index = nan_row["row_index"] + half_window_size
        # usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
        #                                temp_array[row_index - half_window_size:row_index]])
        return temp_usage_mean.loc[nan_row["hour"]].usage
        # return np.nanmean(usage_window).sum()

    temp_df["row_index"] = temp_df.index
    filled_nan = temp_df[temp_nan_index].swifter.progress_bar(False). \
        apply(inner_window_filler, axis=1).to_numpy().reshape(-1, 1)

    return pd.Series([filled_nan, final_temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset("0.01")
    x = x[x.id == 55]
    x_nan = x_nan[x_nan.id == 55]
    filled_users = x_nan.groupby("id").apply(fill_nan, 4)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
