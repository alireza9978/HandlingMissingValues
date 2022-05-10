import kmedoids
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

from src.methods.BaseModel.Base import Base
from src.preprocessing.smart_star.load_dataset import get_train_test_fully_modified_date


class Kmedoids(Base):

    def train_test_save(self, nan_percent_value):
        super().train(Kmedoids.get_train_params(), Kmedoids.fill_nan)
        super().test(Kmedoids.get_train_params(), Kmedoids.fill_nan_test)
        super().save_result(Kmedoids.get_name(), nan_percent_value)

    @staticmethod
    def fill_nan(temp_df: pd.DataFrame, medoids):
        user_id = temp_df["id"].values[0]
        x = temp_df.drop(columns=["id", "usage"]).to_numpy()

        calculated_distances = pairwise_distances(x)
        temp_result = kmedoids.fasterpam(calculated_distances, medoids)
        # impute missing values with mean value of each cluster
        temp_df["cluster_label"] = temp_result.labels
        clusters_mean = temp_df[~temp_df["usage"].isna()].groupby("cluster_label").agg({"usage": "mean"})

        nan_row = temp_df[temp_df["usage"].isna()].drop(columns=["usage"])
        temp_nan_index = nan_row.index.to_numpy()

        filled_nan = nan_row.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]
        return pd.DataFrame({"predicted_usage": filled_nan.to_numpy().squeeze()},
                            index=temp_nan_index.squeeze()), user_id, None

    @staticmethod
    def fill_nan_test(temp_df, other_input):
        _, train_param = other_input
        result, _, _ = Kmedoids.fill_nan(temp_df, train_param)
        return result

    @staticmethod
    def get_train_params():
        return [2, 3, 4, 5, 6, 7, 8, 9, 10]

    @staticmethod
    def get_name():
        return "kmedoids"


if __name__ == '__main__':
    nan_percent = "0.01"
    model = Kmedoids(get_train_test_fully_modified_date(nan_percent, 0.3))
    model.train_test_save(nan_percent)
