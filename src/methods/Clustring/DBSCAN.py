import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_fully_modified_date


class Dbscan(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Dbscan.get_train_params(), Dbscan.fill_nan)
        super().test(Dbscan.get_train_params(), Dbscan.fill_nan_test)
        super().save_result(Dbscan.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, train_param):
        eps, min_samples = train_param
        user_id = temp_df["id"].values[0]

        temp_df = temp_df.drop(columns=["id"])
        nan_index = temp_df[temp_df["usage"].isna()].index.to_numpy()
        x_train = temp_df.drop(columns=["usage"]).to_numpy()

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)

        clu = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = clu.fit_predict(x_train)

        # impute missing values with mean value of each cluster
        temp_df.insert(0, "cluster_label", y_pred, True)
        complete_row = temp_df[~temp_df["usage"].isna()]
        clusters_mean = complete_row.groupby("cluster_label").agg({"usage": "mean"})
        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["usage"])

        filled_nan = nan_row.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()},
                            index=nan_index.squeeze()), user_id, scaler

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        eps, min_samples = train_param
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        scaler = self.params[str(train_param)][user_id]

        nan_index = temp_df[temp_df["usage"].isna()].index.to_numpy()

        x_train = temp_df.drop(columns=["usage"]).to_numpy()
        x_train = scaler.transform(x_train)

        clu = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = clu.fit_predict(x_train)

        # impute missing values with mean value of each cluster
        temp_df.insert(0, "cluster_label", y_pred, True)
        complete_row = temp_df[~temp_df["usage"].isna()]
        clusters_mean = complete_row.groupby("cluster_label").agg({"usage": "mean"})
        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["usage"])

        filled_nan = nan_row.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()}, index=nan_index.squeeze())

    @staticmethod
    def get_train_params():
        # todo set params
        return [(0.5, 5), (10, 50)]

    @staticmethod
    def get_name():
        return "dbscan"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Dbscan(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
