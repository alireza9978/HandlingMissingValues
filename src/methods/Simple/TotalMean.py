import numpy as np
import pandas as pd

from src.methods.BaseModel.Base import Base
from src.preprocessing.smart_star.load_dataset import get_train_test_dataset


class TotalMean(Base):

    def train_test_save(self, nan_percent_value):
        super().train(TotalMean.get_train_params(), TotalMean.fill_nan)
        super().test(TotalMean.get_train_params(), TotalMean.fill_nan_test)
        super().save_result(TotalMean.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df, _):
        user_id = temp_df["id"].values[0]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_mean = np.nanmean(temp_array).sum()
        filled_nan = np.nan_to_num(temp_array, nan=temp_mean)
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index].squeeze()},
                            index=final_filled_nan_index.squeeze()), user_id, temp_mean

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        param = self.params[str(train_param)][user_id]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        filled_nan = np.nan_to_num(temp_array, nan=param)
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index].squeeze()},
                            index=final_filled_nan_index.squeeze())

    @staticmethod
    def get_name():
        return "total_mean"

    @staticmethod
    def get_train_params():
        return ["none"]


if __name__ == '__main__':
    nan_percent = "0.01"
    model = TotalMean(get_train_test_dataset(nan_percent, 0.3))
    model.train_test_save(nan_percent)
