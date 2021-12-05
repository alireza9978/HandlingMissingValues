import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MinMaxScaler

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_fully_modified_date


class LinearRegressionImputer(Base):

    def train_test_save(self, nan_percent_value):
        super().train(LinearRegressionImputer.get_train_params(), LinearRegressionImputer.fill_nan)
        super().test(LinearRegression.get_train_params(), LinearRegression.fill_nan_test)
        # super().save_result(LinearRegression.get_name(), nan_percent_value)

    @staticmethod
    def get_train_params():
        return [4, 8, 12, 24]

    @staticmethod
    def get_name():
        return "linear_regression"

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, data_points_count):
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])

        # getting the nan indexes
        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        non_nan_rows = temp_df.drop(index=nan_index)

        x_train = non_nan_rows.drop(columns=["usage"]).to_numpy()
        y_train = non_nan_rows["usage"].to_numpy().reshape(-1, 1)
        x_test = nan_row.drop(columns=["usage"]).to_numpy()

        # scale input values
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)

        all_calculated_distances = euclidean_distances(x_test, x_train)
        nearest_usage_indexes = np.argsort(all_calculated_distances)[:, :data_points_count]
        x_train = x_train[nearest_usage_indexes]
        y_train = y_train[nearest_usage_indexes]

        result = []
        for x, y, test_x in zip(x_train, y_train, x_test):
            reg = LinearRegression()
            reg = reg.fit(x, y)
            result.append(reg.predict(test_x.reshape(1, test_x.shape[0])).squeeze())

        result = np.array(result).reshape(-1, 1)
        result = y_scaler.inverse_transform(result)
        return pd.DataFrame({"predicted_usage": result}, index=nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = LinearRegressionImputer.fill_nan(temp_df, train_param)
        return result


if __name__ == '__main__':
    nan_percent = "0.01"
    model = LinearRegressionImputer(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
