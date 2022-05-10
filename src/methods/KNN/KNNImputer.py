import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from src.methods.BaseModel.Base import Base
from src.preprocessing.smart_star.load_dataset import get_train_test_fully_modified_date


class Knn(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Knn.get_train_params(), Knn.fill_nan)
        super().test(Knn.get_train_params(), Knn.fill_nan_test)
        super().save_result(Knn.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, n_neighbors):
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])

        # getting the nan indexes
        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        non_nan_rows = temp_df.drop(index=nan_index)

        # scale input values
        scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        x_train = scaler.fit_transform(non_nan_rows.drop(columns=["usage"]))
        x_test = scaler.transform(nan_row.drop(columns=["usage"]))
        y_train = y_scaler.fit_transform(non_nan_rows["usage"].to_numpy().reshape(-1, 1))
        x_train = np.hstack((x_train, y_train))
        temp_nan_array = np.empty((x_test.shape[0], 1))
        temp_nan_array[:] = np.NAN
        x_test = np.hstack((x_test, temp_nan_array))

        # define imputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance', metric='nan_euclidean')
        # fit on the dataset
        imputer.fit(x_train)
        # fill nans in train data
        filled_nan = imputer.transform(x_test)
        filled_nan = y_scaler.inverse_transform(filled_nan[:, -1].reshape(-1, 1))
        return pd.DataFrame({"predicted_usage": filled_nan.squeeze()},
                            index=nan_index.squeeze()), user_id, (scaler, y_scaler, imputer)

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler, y_scaler, imputer = self.params[str(train_param)][user_id]

        nan_row = temp_df[temp_df["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        x_test = scaler.transform(nan_row.drop(columns=["usage"]))
        temp_nan_array = np.empty((x_test.shape[0], 1))
        temp_nan_array[:] = np.NAN
        x_test = np.hstack((x_test, temp_nan_array))

        filled_nan = imputer.transform(x_test)
        filled_nan = y_scaler.inverse_transform(filled_nan[:, -1].reshape(-1, 1))
        return pd.DataFrame({"predicted_usage": filled_nan.squeeze()}, index=nan_index.squeeze())

    @staticmethod
    def get_name():
        return "knn"

    @staticmethod
    def get_train_params():
        return [12, 24, 72]


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Knn(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
