import numpy as np
import pandas as pd

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class Interpolation(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Interpolation.get_train_params(), Interpolation.fill_nan)
        super().test(Interpolation.get_train_params(), Interpolation.fill_nan_test)
        super().save_result(Interpolation.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, train_param):
        user_id = temp_df["id"].values[0]

        temp_array = temp_df.usage.to_numpy().reshape(-1, 1).copy()
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        df = pd.DataFrame(temp_array)
        df.columns = ["data"]
        df = df.interpolate(method=train_param, order=2)
        filled_nan = df["data"].to_numpy().round(5)
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index]},
                            index=final_filled_nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = Interpolation.fill_nan(temp_df, train_param)
        return result

    @staticmethod
    def get_name():
        return "interpolation"

    @staticmethod
    def get_train_params():
        return ["linear", "spline", "polynomial"]


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Interpolation(get_train_test_dataset(nan_percent, 0.3))
    model.train_test_save(nan_percent)
