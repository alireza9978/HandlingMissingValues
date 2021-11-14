import pandas as pd
from sklearn.cluster import KMeans

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def fill_nan(temp_df: pd.DataFrame, n_clusters):
    nan_row = temp_df[temp_df["usage"].isna()]
    temp_nan_index = nan_row.index.to_numpy()
    complete_row = temp_df[~temp_df["usage"].isna()]

    x_train = complete_row.drop(columns=["id", "usage"]).to_numpy()
    x_test = nan_row.drop(columns=["id", "usage"]).to_numpy()
    # clustering all complete rows
    clu = KMeans(n_clusters=n_clusters)
    clu = clu.fit(x_train)
    # predicting each row's cluster based on not nan rows with trained model
    y_test = clu.predict(x_test)
    y_test = pd.DataFrame(y_test, columns=["cluster_label"])

    # impute missing values with mean value of each cluster
    complete_row.insert(0, "cluster_label", clu.labels_, True)
    clusters_mean = complete_row.groupby("cluster_label").agg({"usage": "mean"})
    filled_nan = y_test.set_index("cluster_label").join(clusters_mean).reset_index()["usage"]

    return pd.Series([filled_nan.to_numpy().reshape(-1, 1), temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    for i in range(2, 10):
        filled_users = apply_parallel(x_nan.groupby("id"), fill_nan, i)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        print(i, evaluate_dataframe(filled_users, mean_square_error))
