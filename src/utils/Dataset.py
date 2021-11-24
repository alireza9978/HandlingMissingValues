import numpy as np
import pandas as pd

from src.preprocessing.load_dataset import root
from src.utils.Methods import method_name_single_feature, method_name_single_feature_param, \
    method_single_feature_param_value


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
    temp_df = temp_df.reset_index()
    temp_df.to_csv(path, index=False)


def load_error(nan_percent: str, method_name: str, measure: str, params=None):
    if params is not None:
        method_name += str(params)
    path = root + f"results/errors/error_df_method_{method_name}_nan_{nan_percent}_{measure}.csv"
    return pd.read_csv(path)


def get_all_error_dfs(nan_percent, measure_name):
    methods_name = []
    method_df = []
    error_df = pd.DataFrame()
    for name in method_name_single_feature:
        method_df.append(load_error(nan_percent, name, measure_name))
        methods_name.append(name)
    for name, params in zip(method_name_single_feature_param, method_single_feature_param_value):
        best_df = None
        best_error = None
        for param in params:
            temp_error_df = load_error(nan_percent, name, measure_name, param)
            temp_error = temp_error_df["error"].mean()
            if best_df is None:
                best_df = temp_error_df
                best_error = temp_error
            else:
                if temp_error < best_error:
                    best_df = temp_error_df
                    best_error = temp_error
        method_df.append(best_df)
        methods_name.append(name)

    for name, temp_df in zip(methods_name, method_df):
        error_df[name] = temp_df.set_index("index")["error"]

    return error_df.dropna()
