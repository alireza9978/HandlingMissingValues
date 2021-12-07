import warnings

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.measurements.Measurements import evaluate_dataframe_two
from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class Arima(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Arima.get_train_params(), Arima.fill_nan)
        self.test(Arima.get_train_params(), Arima.fill_nan_test)
        super().save_result(Arima.get_name(), nan_percent_value)

    def test(self, train_params, method):
        from src.utils.Methods import measures, measures_name

        self.test_error_dfs = {}
        self.test_errors = pd.DataFrame()

        for train_param in train_params:
            temp_test_result = self.test_nan_df.groupby("id").apply(method, (self, train_param))
            temp_test_result = temp_test_result.reset_index(level=0).join(self.test_df[["usage"]])
            temp_result_list = [str(train_param)]
            for measure, measure_name in zip(measures, measures_name):
                error, temp_test_error_df = evaluate_dataframe_two(temp_test_result.copy(), measure)
                self.test_error_dfs[str(train_param) + "_" + str(measure_name)] = temp_test_error_df
                temp_result_list.append(error)
            self.test_errors = self.test_errors.append(pd.Series(temp_result_list), ignore_index=True)

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
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            user_id = temp_df["id"].values[0]
            temp_df = temp_df.drop(columns=["id"]).copy()
            scaler, arima_model_fitted = self.params[str(train_param)][user_id]
            nan_row = temp_df[temp_df["usage"].isna()]
            nan_index = nan_row.index.to_numpy()
            temp_df['usage'] = pd.Series(arima_model_fitted.forecast(temp_df["usage"].shape[0]).copy().values,
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
    model.train_test_save(nan_percent)
