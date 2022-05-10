import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.preprocessing.smart_star.load_dataset import get_dataset_auto, root
from src.utils.Dataset import get_all_error_dfs
from src.utils.Methods import measures_name


def get_data_set(user_id):
    nan_percent = "0.1"
    df = get_dataset_auto(nan_percent)
    df = df[df.id == user_id]
    df = df[df.usage.isna()].drop(columns=["usage"])
    predicted_usage_df = get_all_error_dfs(nan_percent, measures_name[0], "predicted_usage")
    predicted_usage_df = predicted_usage_df.loc[df.index].dropna()

    error_usage_df = get_all_error_dfs(nan_percent, measures_name[0])
    error_usage_df = error_usage_df.loc[df.index].dropna()

    y = df["real_usage"][predicted_usage_df.index].to_numpy()
    x = predicted_usage_df.to_numpy()

    temp_results_dict = {}
    for index, col_name in enumerate(predicted_usage_df.columns):
        temp_results_dict[col_name] = mean_squared_error(y, x[:, index])

    temp_results_dict["best possible"] = error_usage_df.min(axis=1).mean()

    temp_train_x, temp_test_x, temp_train_y, temp_test_y = train_test_split(x, y, train_size=0.7, random_state=44)

    temp_x_scaler = StandardScaler()
    temp_train_x = temp_x_scaler.fit_transform(temp_train_x)
    temp_test_x = temp_x_scaler.transform(temp_test_x)

    temp_y_scaler = StandardScaler()
    temp_train_y = temp_y_scaler.fit_transform(temp_train_y.reshape(-1, 1)).squeeze()

    return temp_train_x, temp_test_x, temp_train_y, temp_test_y, temp_x_scaler, temp_y_scaler, temp_results_dict


if __name__ == '__main__':
    results_df = pd.DataFrame()
    for i in range(1, 115):
        train_x, test_x, train_y, test_y, x_scaler, y_scaler, results_dict = get_data_set(i)

        reg = SVR(kernel="linear")
        # reg = linear_model.Ridge(alpha=.5)
        # reg = LassoCV(random_state=42)
        # reg = SGDRegressor(max_iter=1000, tol=1e-3)
        reg.fit(train_x, train_y)
        y_hat = reg.predict(test_x)
        y_hat = y_scaler.inverse_transform(y_hat.reshape(-1, 1)).squeeze()
        results_dict["novel approach"] = mean_squared_error(test_y, y_hat)
        results_df = results_df.append(pd.Series(results_dict, name=i))

    path = root + f"results/combination.csv"
    results_df = results_df.reset_index()
    results_df.to_csv(path, index=True)

    # print("best mse for single method: ", method_results[method_results.argmin()],
    #       method_results.index[method_results.argmin()])
    # print("train: ")
    # print("accuracy: ", accuracy_score(train_y, train_prediction))
    # print("mse: ", calculate_error(train_error_df, train_prediction))
    # print("test: ")
    # print("accuracy: ", accuracy_score(test_y, y_prediction))
    # print("mse: ", calculate_error(test_error_df, y_prediction))
