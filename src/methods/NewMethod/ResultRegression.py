import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.methods.BaseModel.Base import Base
from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_train_test_dataset
from src.utils.Methods import all_methods


def load_all_errors():
    final_train = pd.DataFrame()
    final_test = pd.DataFrame()
    for nan_percent in nan_percents_str[:1]:
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
    return final_train, final_test


def load_all_error_dfs(nan_percent):
    final_train = None
    final_test = None
    for model in all_methods:
        temp_train, temp_test = Base.load_error_dfs(model.get_name(), nan_percent, "mse", model.get_train_params())
        if final_train is None:
            final_test = temp_test
            final_train = temp_train
        else:
            final_train = final_train.join(temp_train)
            final_test = final_test.join(temp_test)
    data_frames = get_train_test_dataset(nan_percent, 0.3)
    final_test = final_test.join(data_frames[1][["usage"]])
    final_train = final_train.join(data_frames[0][["usage"]])
    return final_train, final_test


def train_regression(train_error_dfs, test_error_dfs, train_errors, test_errors):
    train_error_dfs = train_error_dfs.dropna()
    test_error_dfs = test_error_dfs.dropna()

    x_scaler = MinMaxScaler()
    x_train = train_error_dfs.drop(columns=["usage"]).to_numpy()
    x_test = test_error_dfs.drop(columns=["usage"]).to_numpy()
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    y_scaler = MinMaxScaler()
    y_train = train_error_dfs[["usage"]].to_numpy()
    y_test = test_error_dfs[["usage"]].to_numpy()
    y_train = y_scaler.fit_transform(y_train)

    reg = LinearRegression()
    reg = reg.fit(x_train, y_train)
    y_pred = reg.predict(x_train)
    y_pred = y_scaler.inverse_transform(y_pred)
    print("regression train = ", mean_squared_error(y_train, y_pred))
    y_pred = reg.predict(x_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    print("regression test = ", mean_squared_error(y_test, y_pred))
    print("best method in train ", train_errors.iloc[train_errors.mse.argmin()].mse)
    print("best method in test ", test_errors.iloc[test_errors.mse.argmin()].mse)


if __name__ == '__main__':
    main_train_errors, main_test_errors = load_all_errors()
    main_train_error_dfs, main_test_error_dfs = load_all_error_dfs("0.01")
    train_regression(main_train_error_dfs, main_test_error_dfs, main_train_errors, main_test_errors)
