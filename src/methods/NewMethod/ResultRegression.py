import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error

from src.preprocessing.air.load_dataset import load_air_dfs
from src.utils.Dataset import load_all_error_dfs, load_all_methods_result


def train_model(train_x, test_x, train_y, test_y):
    # x_scaler = MinMaxScaler()
    # train_x = x_scaler.fit_transform(train_x)
    # test_x = x_scaler.transform(test_x)

    # y_scaler = MinMaxScaler()
    # scaled_train_y = y_scaler.fit_transform(train_y)

    models = [
        LinearRegression(), Lasso(), SGDRegressor(), ElasticNet(), Ridge(),
        RandomForestRegressor(),
    ]
    # BayesianRidge(), PassiveAggressiveRegressor()]

    train_results = []
    results = []
    for model in models:
        # model = model.fit(train_x, scaled_train_y.ravel())
        # y_pred_train = y_scaler.inverse_transform(model.predict(train_x).reshape(-1, 1))
        # y_pred_test = y_scaler.inverse_transform(model.predict(test_x).reshape(-1, 1))

        model = model.fit(train_x, train_y)
        y_pred_train = model.predict(train_x).reshape(-1, 1)
        y_pred_test = model.predict(test_x).reshape(-1, 1)

        train_results.append(mean_absolute_error(train_y, y_pred_train))
        results.append(mean_absolute_error(test_y, y_pred_test))
    return train_results, results


def print_best_method(user_train_error_dfs, user_test_error_dfs, best_methods):
    best_error = 1000
    best_error_col = None
    best_error_test = 1000
    best_error_col_test = None
    y_true = user_train_error_dfs["usage"]
    y_true_test = user_test_error_dfs["usage"]
    for col in best_methods:
        temp_error = mean_absolute_error(y_true, user_train_error_dfs[col])
        if temp_error < best_error:
            best_error = temp_error
            best_error_col = col
        temp_error = mean_absolute_error(y_true_test, user_test_error_dfs[col])
        if temp_error < best_error_test:
            best_error_test = temp_error
            best_error_col_test = col
    return [best_error, best_error_col, best_error_test, best_error_col_test]


def train_regression(train_error_dfs, test_error_dfs, train_errors, test_errors):
    train_error_dfs = train_error_dfs.dropna()
    test_error_dfs = test_error_dfs.dropna()
    best_methods = train_errors.sort_values("mae").apply(lambda row: row["model"] + "_" + str(row["params"]),
                                                         axis=1)[:15].to_list()
    required_columns = best_methods + ["usage", "id"]
    train_error_dfs = train_error_dfs[required_columns]
    test_error_dfs = test_error_dfs[required_columns]

    train_results_df = pd.DataFrame()
    test_results_df = pd.DataFrame()
    # results_methods = pd.DataFrame()
    # for user_id in train_error_dfs["id"].unique():
    # user_train_error_dfs = train_error_dfs[train_error_dfs.id == user_id].drop(columns=["id"])
    # user_test_error_dfs = test_error_dfs[test_error_dfs.id == user_id].drop(columns=["id"])
    user_train_error_dfs = train_error_dfs.drop(columns=["id"])
    user_test_error_dfs = test_error_dfs.drop(columns=["id"])
    x_train = user_train_error_dfs.drop(columns=["usage"]).to_numpy()
    x_test = user_test_error_dfs.drop(columns=["usage"]).to_numpy()

    y_train = user_train_error_dfs[["usage"]].to_numpy()
    y_test = user_test_error_dfs[["usage"]].to_numpy()

    x_train = x_train[:int(x_train.shape[0] / 2)]
    y_train = y_train[:int(y_train.shape[0] / 2)]

    train_result, test_result = train_model(x_train, x_test, y_train, y_test)
    test_results_df = test_results_df.append(pd.Series(test_result), ignore_index=True)
    train_results_df = train_results_df.append(pd.Series(train_result), ignore_index=True)
    # b = print_best_method(user_train_error_dfs, user_test_error_dfs, best_methods)
    # results_methods = results_methods.append(pd.Series(b), ignore_index=True)

    print("best train mae = ")
    print(train_results_df)
    print("best test mae = ")
    print(test_results_df)
    print(print_best_method(train_error_dfs, test_error_dfs, best_methods))


if __name__ == '__main__':
    errors = load_all_methods_result(["air"])
    error_dfs = load_all_error_dfs("air", load_air_dfs()[0])
    for error, error_df in zip(errors, error_dfs):
        main_train_errors, main_test_errors = error
        main_train_error_dfs, main_test_error_dfs = error_df
        train_regression(main_train_error_dfs, main_test_error_dfs, main_train_errors, main_test_errors)
