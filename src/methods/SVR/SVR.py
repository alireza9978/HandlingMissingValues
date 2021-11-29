import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_train_test_fully_modified_date
from src.utils.parallelizem import apply_parallel_two, apply_parallel


class Svr:

    def __init__(self, dfs):
        # dfs = list(dfs)
        # for i in range(len(dfs)):
        #     dfs[i] = dfs[i][dfs[i].id == 100]
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
        output = apply_parallel_two(self.train_nan_df.groupby("id"), Svr.fill_nan)
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
        self.test_result = apply_parallel(self.test_nan_df.groupby("id"), Svr.fill_nan_test, self)
        self.test_result = self.test_result.join(self.test_df[["usage"]])
        error, self.test_error_df = evaluate_dataframe_two(self.test_result, mean_square_error)
        print("test error = ", error)

    @staticmethod
    def get_name():
        return "svr"

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame):
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler = MinMaxScaler()

        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        non_nan_rows = temp_df.drop(index=nan_index)
        svr_model = SVR(C=1000.0, epsilon=0.15, kernel='rbf', gamma='scale', degree=5)
        svr_model.fit(non_nan_rows.drop(columns=['usage']), non_nan_rows['usage'])
        usage = svr_model.predict(nan_row.drop(columns=['usage'])).reshape(-1, 1)
        return pd.DataFrame({"predicted_usage": usage.squeeze()},
                            index=nan_index.squeeze()), user_id, (scaler, svr_model)

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, self):
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler, svr_model = self.params[user_id]

        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        usage = svr_model.predict(nan_row.drop(columns=['usage'])).reshape(-1, 1)
        return pd.DataFrame({"predicted_usage": usage.squeeze()}, index=nan_index.squeeze())


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Svr(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train()
    model.test()
