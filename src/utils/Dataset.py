from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.smart_star.load_dataset import get_train_test_dataset_triple
from src.preprocessing.smart_star.load_dataset import root


# from src.utils.Methods import method_name_single_feature, method_name_single_feature_param, \
#     method_single_feature_param_value


def get_random_user(x: pd.DataFrame, x_nan: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    user_ids = x.id.unique()
    temp_id = np.random.choice(user_ids, 1)[0]
    x = x[x.id == temp_id]
    # x = x.reset_index(drop=True)
    x_nan = x_nan[x_nan.id == temp_id]
    # x_nan = x_nan.reset_index(drop=True)
    return x, x_nan


def get_user_by_id(x: pd.DataFrame, x_nan: pd.DataFrame, temp_id: int) -> (pd.DataFrame, pd.DataFrame):
    x = x[x.id == temp_id]
    x_nan = x_nan[x_nan.id == temp_id]
    # x = x.reset_index(drop=True)
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


def save_error_two(temp_df: pd.DataFrame, nan_percent: str, method_name: str, measure_params, train: bool = True):
    prefix = "train"
    if not train:
        prefix = "test"
    path = root + f"results/errors/{prefix}_error_df_method_{method_name}_nan_{nan_percent}_{measure_params}.csv"
    temp_df.to_csv(Path(path), index=True)


def load_error_two(nan_percent: str, method_name: str, measure_params, train: bool = True):
    prefix = "train"
    if not train:
        prefix = "test"
    path = root + f"results/errors/{prefix}_error_df_method_{method_name}_nan_{nan_percent}_{measure_params}.csv"
    temp_df = pd.read_csv(Path(path), index_col=0)
    return temp_df


def load_all_methods_result(nan_percents_str):
    from src.methods.BaseModel.Base import Base
    from src.utils.Methods import all_methods

    final_train = pd.DataFrame()
    final_test = pd.DataFrame()
    for nan_percent in nan_percents_str:
        middle_train = pd.DataFrame()
        middle_test = pd.DataFrame()
        for model in all_methods:
            temp_train, temp_test = Base.load_errors(model.get_name(), nan_percent)
            temp_train["model"] = model.get_name()
            temp_test["model"] = model.get_name()
            middle_train = pd.concat([middle_train, temp_train])
            middle_test = pd.concat([middle_test, temp_test])
        middle_train["nan_percent"] = nan_percent
        middle_test["nan_percent"] = nan_percent
        final_train = pd.concat([middle_train, final_train])
        final_test = pd.concat([middle_test, final_test])
    return [(final_train, final_test)]


def load_all_error_dfs(nan_percent, data_frames):
    from src.methods.BaseModel.Base import Base
    from src.utils.Methods import all_methods

    final_train = None
    final_test = None
    for model in all_methods:
        temp_train, temp_test = Base.load_error_dfs(model.get_name(), nan_percent, "mae", model.get_train_params())
        if final_train is None:
            final_test = temp_test
            final_train = temp_train
        else:
            final_train = final_train.join(temp_train)
            final_test = final_test.join(temp_test)
    final_test = final_test.join(data_frames[1][["usage", "id"]])
    final_train = final_train.join(data_frames[0][["usage", "id"]])
    return [(final_train, final_test)]


def load_all_errors_triple(nan_percents_str):
    from src.methods.BaseModel.Base import Base
    from src.utils.Methods import methods_trainable

    results = []
    for ch in ["a", "b", "c"]:
        final_train = pd.DataFrame()
        final_test = pd.DataFrame()
        for nan_percent in nan_percents_str[:1]:
            middle_train = pd.DataFrame()
            middle_test = pd.DataFrame()
            for model in methods_trainable:
                temp_train, temp_test = Base.load_errors(model.get_name(), nan_percent + "_" + ch)
                temp_train["model"] = model.get_name()
                temp_test["model"] = model.get_name()
                middle_train = pd.concat([middle_train, temp_train])
                middle_test = pd.concat([middle_test, temp_test])
            middle_train["nan_percent"] = nan_percent
            middle_test["nan_percent"] = nan_percent
            final_train = pd.concat([middle_train, final_train])
            final_test = pd.concat([middle_test, final_test])
        results.append((final_train, final_test))
    return results


def load_all_error_dfs_triple(nan_percent):
    from src.methods.BaseModel.Base import Base
    from src.utils.Methods import methods_trainable

    main_dfs = get_train_test_dataset_triple(nan_percent)
    results = []
    for ch, data_frames in zip(["a", "b", "c"], main_dfs):
        data_frames = data_frames[0]
        final_train = None
        final_test = None
        for model in methods_trainable:
            temp_train, temp_test = Base.load_error_dfs(model.get_name(), nan_percent, "mse" + "_" + ch,
                                                        model.get_train_params())
            if final_train is None:
                final_test = temp_test
                final_train = temp_train
            else:
                final_train = final_train.join(temp_train)
                final_test = final_test.join(temp_test)

        final_test = final_test.join(data_frames[1][["usage", "id"]])
        final_train = final_train.join(data_frames[0][["usage", "id"]])
        results.append((final_train, final_test))
    return results
