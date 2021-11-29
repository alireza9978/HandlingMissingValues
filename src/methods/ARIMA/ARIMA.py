import warnings

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class Arima(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Arima.get_train_params(), Arima.fill_nan)
        super().test(Arima.get_train_params(), Arima.fill_nan_test)
        super().save_result(Arima.get_name(), nan_percent_value)

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
    model.train_test_save(nan_percent)
