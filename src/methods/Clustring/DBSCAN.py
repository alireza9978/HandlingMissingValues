import pandas as pd
from sklearn.cluster import DBSCAN

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def fill_nan(temp_df: pd.DataFrame):
    x_train = temp_df.drop(columns=["id", "usage"]).to_numpy()

    clu = DBSCAN(eps=10, min_samples=50)
    y_train = clu.fit_predict(x_train)
    # impute missing values with mean value of each cluster
    temp_df["cluster_label"] = y_train
    clusters_mean = temp_df[~temp_df["usage"].isna()].groupby("cluster_label").agg({"usage": "mean"})
    nan_row = temp_df[temp_df["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    filled_nan = nan_row[["cluster_label"]].reset_index().set_index("cluster_label"). \
        join(clusters_mean).sort_values("index")["usage"]

    return pd.Series([filled_nan.to_numpy().reshape(-1, 1), temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
