import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.methods.BaseModel.Base import Base
from src.preprocessing.smart_star.load_dataset import get_train_test_fully_modified_date


class Svr(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Svr.get_train_params(), Svr.fill_nan)
        super().test(Svr.get_train_params(), Svr.fill_nan_test)
        super().save_result(Svr.get_name(), nan_percent_value)

    @staticmethod
    def get_train_params():
        return [(1, 2), (1, 3), (1, 5)]
        # return [(1, 2)]

    @staticmethod
    def get_name():
        return "svr"

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, train_param):
        c, degree = train_param

        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])

        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=['usage'])
        nan_index = nan_row.index.to_numpy()
        non_nan_rows = temp_df.drop(index=nan_index)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(non_nan_rows.drop(columns=['usage']))
        x_test = scaler.transform(nan_row)

        # svr_model = SVR(C=c, epsilon=0.15, kernel='poly', gamma='scale', degree=degree)
        svr_model = SVR(C=1, epsilon=0.15, kernel='poly', gamma='scale', degree=degree)
        svr_model.fit(x_train, non_nan_rows['usage'])
        usage = svr_model.predict(x_test)
        return pd.DataFrame({"predicted_usage": usage.squeeze()},
                            index=nan_index.squeeze()), user_id, (scaler, svr_model)

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=['usage'])
        nan_index = nan_row.index.to_numpy()

        scaler, svr_model = self.params[str(train_param)][user_id]
        x_test = scaler.transform(nan_row)

        usage = svr_model.predict(x_test)

        return pd.DataFrame({"predicted_usage": usage.squeeze()}, index=nan_index.squeeze())


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Svr(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
