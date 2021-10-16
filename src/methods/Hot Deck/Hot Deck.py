import multiprocessing

import numpy as np
import pandas as pd
import swifter
from joblib import Parallel, delayed

from src.measurements.Measurements import mean_square_error, evaluate_dataframe
from src.preprocessing.load_dataset import get_dataset_with_modified_date

a = swifter.config


def apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.concat(result_list)


def fill_nan(temp_array: np.ndarray):
    nan_row = temp_array[temp_array["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    complete_row = temp_array[~temp_array["usage"].isna()]

    def get_nearest_usage(row: pd.Series):
        temp_row = row.drop(["id", "usage"]).to_numpy()

        def distance(inner_row: pd.Series):
            temp_usage = inner_row["usage"]
            temp_distance = np.sum(np.square(inner_row.drop(["id", "usage"]).to_numpy() - temp_row))
            return pd.Series([temp_distance, temp_usage])

        calculated_distances = complete_row.swifter.apply(distance, axis=1)
        calculated_distances = calculated_distances.sort_values(by=0)
        return calculated_distances.loc[0][1]

    filled_nan = nan_row.swifter.apply(get_nearest_usage, axis=1)
    return pd.Series([filled_nan, temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_with_modified_date()
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    print("here")
    filled_users[2] = filled_users[1].swifter.apply(lambda idx: x.loc[idx])
    print("here2")
    print(evaluate_dataframe(filled_users, mean_square_error))
    print("here3")
