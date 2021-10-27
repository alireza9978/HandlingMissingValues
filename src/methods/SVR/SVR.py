from sklearn.preprocessing import MinMaxScaler
from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
import pandas as pd
import numpy as np


def fill_nan(user_data: pd.DataFrame):
    user = user_data.copy()
    # getting the nan indexes
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    non_nan_rows = user.drop(user[nan_index])
    model = SVR(C=1.0, epsilon=0.1, kernel='poly', gamma='scale', degree=3)
    model.fit(non_nan_rows.drop(columns=['usage']), non_nan_rows['usage'])
    usage = model.predict(nan_row.drop(columns=['usage']))
    return pd.Series(usage, nan_index)


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date(nan_percent='0.05')
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
