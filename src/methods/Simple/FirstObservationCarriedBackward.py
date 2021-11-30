import numpy as np
import pandas as pd

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class FirstObservationCarriedBackward(Base):

    def train_test_save(self, nan_percent_value):
        super().train(FirstObservationCarriedBackward.get_train_params(), FirstObservationCarriedBackward.fill_nan)
        super().test(FirstObservationCarriedBackward.get_train_params(), FirstObservationCarriedBackward.fill_nan_test)
        super().save_result(FirstObservationCarriedBackward.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, _):
        user_id = temp_df["id"].values[0]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1).copy()
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        temp_df['usage'] = temp_df['usage'].bfill().ffill()
        filled_nan = temp_df["usage"].to_numpy()
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index]},
                            index=final_filled_nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = FirstObservationCarriedBackward.fill_nan(temp_df, train_param)
        return result

    @staticmethod
    def get_train_params():
        return ["none"]

    @staticmethod
    def get_name():
        return "first_observation_carried_backward"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = FirstObservationCarriedBackward(get_train_test_dataset(nan_percent, 0.3))
    model.train_test_save(nan_percent)
