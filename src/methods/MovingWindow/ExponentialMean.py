import numpy as np
import pandas as pd

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_dataset


class ExponentialMean(Base):

    def train_test_save(self, nan_percent_value):
        super().train(ExponentialMean.get_train_params(), ExponentialMean.fill_nan)
        super().test(ExponentialMean.get_train_params(), ExponentialMean.fill_nan_test)
        super().save_result(ExponentialMean.get_name(), nan_percent_value)

    @staticmethod
    def get_name():
        return "moving_window_exponential_mean"

    @staticmethod
    def get_train_params():
        return [4, 6, 8, 10, 12, 24, 48, 126]

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, window_size):
        if window_size is None:
            return None

        import swifter
        _ = swifter.config
        user_id = temp_df["id"].values[0]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_df = temp_df.reset_index(drop=True)
        half_window_size = int(window_size / 2)
        temp_array = np.pad(temp_array.squeeze(), half_window_size, mode="constant", constant_values=np.nan).reshape(-1,
                                                                                                                     1)
        bottom = np.power(np.full(window_size, 2, dtype=np.uint64), np.concatenate(
            [np.arange(half_window_size, 0, -1, dtype=np.uint64),
             np.arange(1, half_window_size + 1, dtype=np.uint64)]), dtype=np.uint64)
        weights = np.divide(np.ones(bottom.shape[0], dtype=np.uint64), bottom)

        def inner_window_filler(nan_row):
            row_index = int(nan_row["row_index"]) + half_window_size
            usage_window = np.concatenate([temp_array[row_index + 1: row_index + 1 + half_window_size],
                                           temp_array[row_index - half_window_size:row_index]])

            not_nan_mask = ~np.isnan(usage_window)
            usage_window = usage_window[not_nan_mask]
            if usage_window.shape[0] > 0:
                weights_sum = np.sum(weights[not_nan_mask.squeeze()])
                return (usage_window.squeeze() * weights[not_nan_mask.squeeze()]).sum() / weights_sum
            return np.nan

        temp_df["row_index"] = temp_df.index
        temp_nan_index = temp_df.usage.isna()

        filled_nan = temp_df[temp_nan_index].apply(inner_window_filler, axis=1).to_numpy().reshape(-1, 1)

        temp_mean = np.nanmean(temp_array).sum()
        filled_nan = np.nan_to_num(filled_nan, nan=temp_mean)
        return pd.DataFrame({"predicted_usage": filled_nan.squeeze()},
                            index=final_temp_nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = ExponentialMean.fill_nan(temp_df, train_param)
        return result


if __name__ == '__main__':
    nan_percent = "0.01"
    model = ExponentialMean(get_train_test_dataset(nan_percent, 0.3))
    model.train_test_save(nan_percent)
