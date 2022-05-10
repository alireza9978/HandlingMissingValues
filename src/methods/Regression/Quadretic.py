import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.smart_star.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def fill_nan(temp_array: np.ndarray):
    nan_row = temp_array[temp_array["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    complete_row = temp_array[~temp_array["usage"].isna()]

    x_train = complete_row.drop(columns=["id", "usage"]).to_numpy()
    y_train = complete_row.usage.to_numpy().reshape(-1, 1)
    degree = 4
    polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polynomial_reg.fit(x_train, y_train)
    filled_nan = polynomial_reg.predict(nan_row.drop(columns=["id", "usage"]).to_numpy())
    return pd.Series([filled_nan, temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
