import warnings

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_train_test_dataset
from src.utils.parallelizem import apply_parallel_two


class Arima:

    def __init__(self, dfs):
        dfs = list(dfs)
        for i in range(len(dfs)):
            dfs[i] = dfs[i][dfs[i].id == 100]
        train_df, test_df, train_nan_df, test_nan_df = dfs
        self.train_df = train_df
        self.test_df = test_df
        self.train_nan_df = train_nan_df
        self.test_nan_df = test_nan_df
        self.params = None
        self.train_result = None
        self.test_error_df = None

    def train(self):
        self.train_result = {}
        self.params = {}
        for train_param in Arima.get_train_params():
            output = apply_parallel_two(self.train_nan_df.groupby("id"), Arima.fill_nan, train_param)
            temp_result = pd.DataFrame()
            temp_params = {}
            for row in output:
                result_df = row[0]
                temp_result = temp_result.append(result_df)
                temp_params[row[1]] = row[2]
            self.train_result[str(train_param)] = temp_result
            self.params[str(train_param)] = temp_params

    def test(self):
        self.test_error_df = {}
        for train_param in Arima.get_train_params():
            temp_test_result = self.test_nan_df.groupby("id").apply(Arima.fill_nan_test, (self, train_param))
            temp_test_result = temp_test_result.reset_index().set_index("level_1")
            temp_test_result = temp_test_result.join(self.test_df[["usage"]])
            error, temp_test_error_df = evaluate_dataframe_two(temp_test_result, mean_square_error)
            self.test_error_df[str(train_param)] = temp_test_error_df
            print(f"test error {train_param} = ", error)

    @staticmethod
    def fill_nan(temp_df, train_param):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            user_id = temp_df["id"].values[0]
            temp_df = temp_df.drop(columns=["id"])
            scaler = MinMaxScaler()

            nan_row = temp_df[temp_df["usage"].isna()]
            nan_index = nan_row.index.to_numpy()
            temp_df["usage"] = scaler.fit_transform(temp_df['usage'].to_numpy().reshape(-1, 1))
            arima_model = ARIMA(temp_df['usage'].copy(), order=train_param)
            arima_model_fitted = arima_model.fit()
            temp_df['usage'] = pd.Series(arima_model_fitted.fittedvalues.copy())
            predictions = scaler.inverse_transform(temp_df['usage'][nan_index].to_numpy().reshape(-1, 1))
            return pd.DataFrame({"predicted_usage": predictions.squeeze()}, index=nan_index.squeeze()), user_id, (
                scaler, arima_model_fitted)

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        self, train_param = other_input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            user_id = temp_df["id"].values[0]
            temp_df = temp_df.drop(columns=["id"])
            scaler, arima_model_fitted = self.params[str(train_param)][user_id]
            nan_row = temp_df[temp_df["usage"].isna()]
            nan_index = nan_row.index.to_numpy()
            temp_df['usage'] = pd.Series(arima_model_fitted.forecast(temp_df["usage"].shape[0]).values,
                                         index=temp_df.index)
            predictions = scaler.inverse_transform(temp_df['usage'][nan_index].to_numpy().reshape(-1, 1))
            return pd.DataFrame({"predicted_usage": predictions.squeeze()}, index=nan_index.squeeze())

    @staticmethod
    def get_name():
        return "arima"

    @staticmethod
    def get_train_params():
        return [(2, 1, 2), (7, 1, 2)]


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Arima(get_train_test_dataset(nan_percent, 0.3))
    model.train()
    model.test()
