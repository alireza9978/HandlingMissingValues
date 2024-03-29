import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.smart_star.load_dataset import get_complete_dataset


def fill_nan(temp_df: pd.DataFrame):
    nan_row = temp_df[temp_df["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    temp_df["only_date"] = temp_df.date.dt.date
    nan_date = temp_df.loc[temp_nan_index]["only_date"].unique()
    nan_df = temp_df[temp_df.only_date.isin(nan_date)]

    def fill_single_day(day_df: pd.DataFrame):
        temp_y = day_df.usage.to_numpy().reshape(-1, 1)
        # temp_x = day_df.drop(columns=["id", "usage", "date", "only_date"]).to_numpy()
        # temp_x = np.array(list(range(day_df.shape[0]))).reshape(-1, 1)
        temp_x = day_df[['day_x', 'day_y']].to_numpy()
        # todo remove some of columns to reduce computational complexity
        # pca_model = PCA()
        # new_temp_x = pca_model.fit_transform(temp_x)

        nan_index = np.isnan(temp_y)
        not_nan_index = ~np.isnan(temp_y)
        y_train = temp_y[not_nan_index]
        x_train = temp_x[not_nan_index.squeeze()]
        x_test = temp_x[nan_index.squeeze()]

        degree = 8
        polynomial_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        polynomial_reg.fit(x_train, y_train)
        pred = polynomial_reg.predict(x_test).squeeze()
        return pred

    filled_nan = nan_df.groupby("only_date").apply(fill_single_day)
    temp_list = []
    for row in filled_nan.to_list():
        if len(row.shape) >= 1:
            for item in row:
                temp_list.append(np.array(item))
        else:
            temp_list.append(row)
    return pd.Series([np.array(temp_list).reshape(-1, 1), temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_complete_dataset("0.15")
    x_nan = x_nan[x_nan.id == 102]
    x = x[x.id == 102]
    # filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users = x_nan.groupby("id").apply(fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
