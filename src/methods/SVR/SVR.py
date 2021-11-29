import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_fully_modified_date
from src.utils.parallelizem import apply_parallel_two, apply_parallel


class Svr(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Svr.get_train_params(), Svr.fill_nan)
        super().test(Svr.get_train_params(), Svr.fill_nan_test)
        super().save_result(Svr.get_name(), nan_percent_value)

    @staticmethod
    def get_train_params():
        return [(1000.0, 0.15)]

    @staticmethod
    def get_name():
        return "svr"

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, train_param):
        c, epsilon = train_param

        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler = MinMaxScaler()

        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        non_nan_rows = temp_df.drop(index=nan_index)
        svr_model = SVR(C=c, epsilon=epsilon, kernel='rbf', gamma='scale', degree=5)
        svr_model.fit(non_nan_rows.drop(columns=['usage']), non_nan_rows['usage'])
        usage = svr_model.predict(nan_row.drop(columns=['usage'])).reshape(-1, 1)
        return pd.DataFrame({"predicted_usage": usage.squeeze()},
                            index=nan_index.squeeze()), user_id, (scaler, svr_model)

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler, svr_model = self.params[str(train_param)][user_id]

        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        usage = svr_model.predict(nan_row.drop(columns=['usage'])).reshape(-1, 1)
        return pd.DataFrame({"predicted_usage": usage.squeeze()}, index=nan_index.squeeze())


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Svr(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
