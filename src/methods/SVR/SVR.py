import pandas as pd
from sklearn.svm import SVR

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_dataset_fully_modified_date


def get_name():
    return "svr"


def fill_nan(user_data: pd.DataFrame):
    user = user_data.copy()
    # getting the nan indexes
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    non_nan_rows = user.drop(index=nan_index)
    model = SVR(C=1000.0, epsilon=0.15, kernel='poly', gamma='scale', degree=5)
    model.fit(non_nan_rows.drop(columns=['usage']), non_nan_rows['usage'])
    usage = model.predict(nan_row.drop(columns=['usage'])).reshape(-1, 1)
    return pd.DataFrame({"predicted_usage": usage.squeeze()},
                        index=nan_index.squeeze())


if __name__ == '__main__':
    from src.utils.Methods import fill_nan as fn
    from src.utils.Dataset import get_random_user

    nan_percent = "0.01"
    x, x_nan = get_dataset_fully_modified_date(nan_percent=nan_percent)
    x, x_nan = get_random_user(x, x_nan)
    filled_users = fn(x, x_nan, fill_nan)
    error, error_df = evaluate_dataframe_two(filled_users, mean_square_error)
    print(error)
    print(error_df)
