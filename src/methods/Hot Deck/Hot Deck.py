import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from src.measurements.Measurements import mean_square_error, evaluate_dataframe
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def fill_nan(temp_array: pd.DataFrame):
    nan_row = temp_array[temp_array["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    complete_row = temp_array[~temp_array["usage"].isna()]

    def get_nearest_usage(row: pd.Series):
        temp_row = row.drop(["id", "usage"]).to_numpy()
        calculated_distances = euclidean_distances(complete_row.drop(columns=["id", "usage"]).to_numpy(),
                                                   temp_row.reshape(1, -1))
        temp_dict = {"distance": calculated_distances.squeeze(), "usage": complete_row.usage}
        calculated_distances = pd.DataFrame(temp_dict)
        calculated_distances = calculated_distances.sort_values(by="distance")
        calculated_distances = calculated_distances.reset_index()
        return calculated_distances.loc[0][1]

    filled_nan = nan_row.apply(get_nearest_usage, axis=1)
    return pd.Series([filled_nan.to_numpy().reshape(-1, 1), temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
