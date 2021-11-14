import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.preprocessing.load_dataset import get_dataset_fully_modified_date, root
from src.utils.parallelizem import apply_parallel


def evaluate_clustering(temp_df: pd.DataFrame, n_clusters):
    user_id = temp_df.id.values[0]
    x_train = temp_df.drop(columns=["id", "usage"]).to_numpy()
    # clustering all complete rows
    clu = KMeans(n_clusters=n_clusters)
    y_pred = clu.fit_predict(x_train)

    return pd.Series(data=silhouette_score(x_train, y_pred), index=[str(n_clusters)], name=user_id)


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.01")
    # x = x[x.id.isin([63, 5, 42])]
    result = pd.DataFrame()
    result["id"] = pd.Series(x.id.unique())
    result = result.set_index("id")
    for i in range(2, 10):
        scores = apply_parallel(x.groupby("id"), evaluate_clustering, i)
        result = result.join(scores)

    result.to_csv(root + "results/clustering_silhouette_score.csv")
