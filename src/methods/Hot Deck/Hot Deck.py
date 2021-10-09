import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error, evaluate_dataframe
from src.preprocessing.load_dataset import get_dataset_with_modified_date


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

        calculated_distances = complete_row.apply(distance, axis=1)
        calculated_distances = calculated_distances.sort_values(by=0)
        return calculated_distances.loc[0][1]
    filled_nan = nan_row.apply(get_nearest_usage, axis=1)
    return pd.Series([filled_nan, temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_with_modified_date()
    filled_users = x_nan.groupby("id").apply(fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
