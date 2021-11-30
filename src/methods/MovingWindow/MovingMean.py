import numpy as np
import pandas as pd

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class MovingMean(Base):

    def train_test_save(self, nan_percent_value):
        super().train(MovingMean.get_train_params(), MovingMean.fill_nan)
        super().test(MovingMean.get_train_params(), MovingMean.fill_nan_test)
        super().save_result(MovingMean.get_name(), nan_percent_value)

    @staticmethod
    def get_name():
        return "moving_window_mean"

    @staticmethod
    def get_train_params():
        return [4, 6, 8, 10, 12, 24, 48, 168, 720]

    @staticmethod
    def fill_nan(temp_df, train_param):
        window_size = train_param
        user_id = temp_df["id"].values[0]
        import swifter
        _ = swifter.config

        final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_df = temp_df.reset_index(drop=True)
        temp_nan_index = temp_df.usage.isna()
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        half_window_size = int(window_size / 2)

        def inner_window_filler(nan_row):
            row_index = nan_row["row_index"]
            usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
                                           temp_array[row_index - half_window_size:row_index]])
            usage_window = usage_window[~np.isnan(usage_window)]
            if usage_window.shape[0] > 0:
                return np.nanmean(usage_window).sum()
            return np.nan

        temp_df["row_index"] = temp_df.index
        filled_nan = temp_df[temp_nan_index].apply(inner_window_filler, axis=1).to_numpy().reshape(-1, 1)

        temp_mean = np.nanmean(temp_array).sum()
        filled_nan = np.nan_to_num(filled_nan, nan=temp_mean)
        return pd.DataFrame({"predicted_usage": filled_nan.squeeze()},
                            index=final_temp_nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = MovingMean.fill_nan(temp_df, train_param)
        return result


if __name__ == '__main__':
    nan_percent = "0.01"
    model = MovingMean(get_train_test_dataset(nan_percent, 0.3))
    model.train_test_save(nan_percent)
