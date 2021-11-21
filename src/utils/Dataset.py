import numpy as np
import pandas as pd

from src.preprocessing.load_dataset import root


def get_random_user(x: pd.DataFrame, x_nan: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    user_ids = x.id.unique()
    temp_id = np.random.choice(user_ids, 1)[0]
    x = x[x.id == temp_id]
    # x = x.reset_index(drop=True)
    x_nan = x_nan[x_nan.id == temp_id]
    # x_nan = x_nan.reset_index(drop=True)
    return x, x_nan


def save_error(temp_df: pd.DataFrame, nan_percent: str, method_name: str, measure: str, params=None):
    if params is not None:
        method_name += str(params)
    path = root + f"results/errors/error_df_method_{method_name}_nan_{nan_percent}_{measure}.csv"
    temp_df.to_csv(path, index=False)


def load_error(nan_percent: str, method_name: str, measure: str, params=None):
    if params is not None:
        method_name += str(params)
    path = root + f"results/errors/error_df_method_{method_name}_nan_{nan_percent}_{measure}.csv"
    return pd.read_csv(path)
