import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def fill_nan(temp_array: np.ndarray):
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
        selected_data_points = complete_row.loc[calculated_distances[:100].index]

        x_train = selected_data_points.drop(columns=["id", "usage"]).to_numpy()
        y_train = selected_data_points.usage.to_numpy().reshape(-1, 1)
        reg = LinearRegression()
        reg = reg.fit(x_train, y_train)
        return reg.predict(temp_row.reshape(1, -1)).sum()

    filled_nan = nan_row.apply(get_nearest_usage, axis=1)
    return pd.Series([filled_nan.to_numpy().reshape(-1, 1), temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    # filled_users = x_nan.groupby("id").apply(fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
