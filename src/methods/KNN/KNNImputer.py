from sklearn.preprocessing import MinMaxScaler
from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    return user, scaler


def fill_nan(user: pd.DataFrame):
    user = user.copy()
    # getting the nan indexes
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    # define imputer
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    # fit on the dataset
    user['usage'] = imputer.fit_transform(user)[:,1] # change 1 to the number of column containing missing values
    filled_nans = user.loc[nan_index]
    return pd.Series([filled_nans['usage'],nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date(nan_percent='0.05')
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))


