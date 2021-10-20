import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_with_modified_date


def apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.DataFrame(result_list)


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
    plt.plot(temp_array.iloc[1000:1100].index, temp_array.iloc[1000:1100].usage)
    plt.plot(temp_array.iloc[1000:1100].index, polynomial_reg.predict(temp_array.drop(columns=["id", "usage"]).iloc[1000:1100]))
    plt.show()

    return pd.Series([filled_nan, temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_with_modified_date()
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
