import pandas as pd
from fcmeans import FCM
from sklearn.preprocessing import MinMaxScaler

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_fully_modified_date


class FuzzyCMean(Base):

    def train_test_save(self, nan_percent_value):
        super().train(FuzzyCMean.get_train_params(), FuzzyCMean.fill_nan)
        super().test(FuzzyCMean.get_train_params(), FuzzyCMean.fill_nan_test)
        super().save_result(FuzzyCMean.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, n_clusters):
        user_id = temp_df["id"].values[0]

        nan_row = temp_df[temp_df["usage"].isna()]
        temp_nan_index = nan_row.index.to_numpy()
        complete_row = temp_df[~temp_df["usage"].isna()]

        x_train = complete_row.drop(columns=["id", "usage"]).to_numpy()
        x_test = nan_row.drop(columns=["id", "usage"]).to_numpy()

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # clustering all complete rows
        fcm = FCM(n_clusters=n_clusters)
        fcm.fit(x_train)
        # predicting each row's cluster based on not nan rows with trained model
        y_train = fcm.predict(x_train)
        y_pred = fcm.predict(x_test)
        y_pred = pd.DataFrame(y_pred, columns=["cluster_label"])

        # impute missing values with mean value of each cluster
        complete_row.insert(0, "cluster_label", y_train, True)
        clusters_mean = complete_row.groupby("cluster_label").agg({"usage": "mean"})

        filled_nan = y_pred.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()},
                            index=temp_nan_index.squeeze()), user_id, (scaler, clusters_mean, fcm)

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        scaler, clusters_mean, fcm = self.params[str(train_param)][user_id]

        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["id", "usage"])
        nan_index = nan_row.index.to_numpy()

        x_test = scaler.transform(nan_row.to_numpy())
        y_pred = fcm.predict(x_test)
        y_pred = pd.DataFrame(y_pred, columns=["cluster_label"])

        filled_nan = y_pred.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()}, index=nan_index.squeeze())

    @staticmethod
    def get_train_params():
        return [2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def get_name():
        return "fuzzy_c_mean"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = FuzzyCMean(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
