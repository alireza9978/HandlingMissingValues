from abc import abstractmethod, ABC

import pandas as pd

from src.measurements.Measurements import evaluate_dataframe_two
from src.utils.parallelizem import apply_parallel_two, apply_parallel


class Base(ABC):

    def __init__(self, data_frames):
        data_frames = self.select_one_user(99, data_frames)
        train_df, test_df, train_nan_df, test_nan_df = data_frames
        self.train_df = train_df
        self.test_df = test_df
        self.train_nan_df = train_nan_df
        self.test_nan_df = test_nan_df

        self.params = None

        self.train_error_dfs = None
        self.train_errors = None
        self.test_error_dfs = None
        self.test_errors = None

    @staticmethod
    def select_one_user(user_id, data_frames):
        data_frames = list(data_frames)
        for i in range(len(data_frames)):
            data_frames[i] = data_frames[i][data_frames[i].id == user_id]
        return data_frames

    def train(self, train_params, method):
        from src.utils.Methods import measures, measures_name

        self.train_error_dfs = {}
        self.train_errors = pd.DataFrame()
        self.params = {}
        for train_param in train_params:
            output = apply_parallel_two(self.train_nan_df.groupby("id"), method, train_param)
            temp_result = pd.DataFrame()
            temp_params = {}
            for row in output:
                result_df = row[0]
                temp_result = temp_result.append(result_df)
                temp_params[row[1]] = row[2]

            temp_result = temp_result.join(self.train_df[["usage"]])
            temp_result_list = [str(train_param)]
            for measure, measure_name in zip(measures, measures_name):
                error, temp_train_error_df = evaluate_dataframe_two(temp_result.copy(), measure)
                self.train_error_dfs[str(train_param) + "_" + str(measure_name)] = temp_train_error_df
                temp_result_list.append(error)
            self.train_errors = self.train_errors.append(pd.Series(temp_result_list), ignore_index=True)
            self.params[str(train_param)] = temp_params

    def test(self, train_params, method):
        from src.utils.Methods import measures, measures_name

        self.test_error_dfs = {}
        self.test_errors = pd.DataFrame()

        for train_param in train_params:
            temp_test_result = apply_parallel(self.test_nan_df.groupby("id"), method, (self, train_param))
            temp_test_result = temp_test_result.join(self.test_df[["usage"]])
            temp_result_list = [str(train_param)]
            for measure, measure_name in zip(measures, measures_name):
                error, temp_test_error_df = evaluate_dataframe_two(temp_test_result.copy(), measure)
                self.test_error_dfs[str(train_param) + "_" + str(measure_name)] = temp_test_error_df
                temp_result_list.append(error)
            self.test_errors = self.test_errors.append(pd.Series(temp_result_list), ignore_index=True)

    @abstractmethod
    def train_test_save(self, nan_percent_value):
        pass

    def save_result(self, name, nan_percent):
        from src.preprocessing.load_dataset import root
        from src.utils.Dataset import save_error_two
        from src.utils.Methods import measures_name
        for train_param in self.train_error_dfs.keys():
            error_df = self.train_error_dfs[train_param]
            save_error_two(error_df, nan_percent, name, train_param, train=True)
        for train_param in self.test_error_dfs.keys():
            error_df = self.test_error_dfs[train_param]
            save_error_two(error_df, nan_percent, name, train_param, train=False)
        temp_columns = ["params"] + measures_name
        self.test_errors.columns = temp_columns
        self.test_errors.to_csv(root + f"results/methods/test_result_{name}_{nan_percent}.csv", index=False)
        self.train_errors.columns = temp_columns
        self.train_errors.to_csv(root + f"results/methods/train_result_{name}_{nan_percent}.csv", index=False)
