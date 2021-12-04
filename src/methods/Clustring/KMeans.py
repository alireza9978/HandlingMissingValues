import pandas as pd
from sklearn.cluster import KMeans

from src.methods.BaseModel.Base import Base
from src.preprocessing.load_dataset import get_train_test_fully_modified_date


class Kmeans(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Kmeans.get_train_params(), Kmeans.fill_nan)
        super().test(Kmeans.get_train_params(), Kmeans.fill_nan_test)
        super().save_result(Kmeans.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, n_clusters):
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])

        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["usage"])
        temp_nan_index = nan_row.index.to_numpy()
        complete_row = temp_df[~temp_df["usage"].isna()]

        x_train = complete_row.drop(columns=["usage"]).to_numpy()
        # clustering all complete rows
        clu = KMeans(n_clusters=n_clusters)
        clu = clu.fit(x_train)

        # impute missing values with mean value of each cluster
        complete_row.insert(0, "cluster_label", clu.labels_, True)
        clusters_mean = complete_row.groupby("cluster_label").agg({"usage": "mean"})

        y_pred = clu.predict(nan_row.to_numpy())
        nan_row["cluster_label"] = pd.Series(y_pred, index=nan_row.index)
        filled_nan = nan_row.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()},
                            index=temp_nan_index.squeeze()), user_id, (clu, clusters_mean)

    @staticmethod
    def fill_nan_test(temp_df: pd.DataFrame, other_input):
        self, train_param = other_input
        user_id = temp_df["id"].values[0]
        temp_df = temp_df.drop(columns=["id"])
        clu, clusters_mean = self.params[str(train_param)][user_id]

        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["usage"])
        nan_index = nan_row.index.to_numpy()
        y_pred = clu.predict(nan_row.to_numpy())
        nan_row["cluster_label"] = pd.Series(y_pred, index=nan_row.index)
        filled_nan = nan_row.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()}, index=nan_index.squeeze())

    @staticmethod
    def get_train_params():
        return [2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def get_name():
        return "kmeans"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Kmeans(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
