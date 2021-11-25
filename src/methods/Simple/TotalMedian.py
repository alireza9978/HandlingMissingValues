import numpy as np
import pandas as pd

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_train_test_dataset
from src.utils.parallelizem import apply_parallel, apply_parallel_two


class TotalMedian:

    def __init__(self, dfs):
        train_df, test_df, train_nan_df, test_nan_df = dfs
        self.train_df = train_df
        self.test_df = test_df
        self.train_nan_df = train_nan_df
        self.test_nan_df = test_nan_df
        self.params = None
        self.train_result = None
        self.train_error_df = None
        self.test_result = None
        self.test_error_df = None

    def train(self):
        output = apply_parallel_two(self.train_nan_df.groupby("id"), TotalMedian.fill_nan)
        self.train_result = pd.DataFrame()
        self.params = {}
        for row in output:
            result_df = row[0]
            self.train_result = self.train_result.append(result_df)
            self.params[row[1]] = row[2]
        self.train_result = self.train_result.join(self.train_df[["usage"]])
        error, self.train_error_df = evaluate_dataframe_two(self.train_result, mean_square_error)
        print("train error = ", error)

    def test(self):
        self.test_result = apply_parallel(self.test_nan_df.groupby("id"), TotalMedian.fill_nan_test, self)
        self.test_result = self.test_result.join(self.test_df[["usage"]])
        error, self.test_error_df = evaluate_dataframe_two(self.test_result, mean_square_error)
        print("test error = ", error)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame):
        user_id = temp_df["id"].values[0]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        temp_mean = np.nanmedian(temp_array).sum()
        filled_nan = np.nan_to_num(temp_array, nan=temp_mean)
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index].squeeze()},
                            index=final_filled_nan_index.squeeze()), user_id, temp_mean

    @staticmethod
    def fill_nan_test(temp_df, self):
        user_id = temp_df["id"].values[0]
        param = self.params[user_id]
        temp_array = temp_df.usage.to_numpy().reshape(-1, 1)
        final_filled_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
        filled_nan = np.nan_to_num(temp_array, nan=param)
        temp_nan_index = np.where(np.isnan(temp_array))[0]
        return pd.DataFrame({"predicted_usage": filled_nan[temp_nan_index].squeeze()},
                            index=final_filled_nan_index.squeeze())

    @staticmethod
    def get_name():
        return "total_median"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = TotalMedian(get_train_test_dataset(nan_percent, 0.3))
    model.train()
    model.test()
