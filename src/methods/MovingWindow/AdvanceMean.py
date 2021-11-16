import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_date_modified


def weighted_mean_squared_error(weights, *args):
    train_x = args[0]
    train_y = args[1]
    y_pred = np.expand_dims((train_x * weights).mean(1), 1)
    error = mean_squared_error(train_y, y_pred)
    return error


def predict(weights, test_x):
    y_pred = np.expand_dims((test_x * weights).mean(1), 1)
    return y_pred


def fill_nan(temp_df: pd.DataFrame, k=20):
    import swifter
    _ = swifter.config
    final_temp_nan_index = temp_df.index[temp_df.usage.isna()].to_numpy()
    temp_df = temp_df.reset_index(drop=True)

    train_y = temp_df.loc[~temp_df.usage.isna(), "usage"].to_numpy().reshape(-1, 1)
    train_x = temp_df.drop(columns=["usage", "id"]).loc[~temp_df.usage.isna()].to_numpy()
    test_x = temp_df.drop(columns=["usage", "id"]).loc[temp_df.usage.isna()].to_numpy()

    pairwise_distance = euclidean_distances(train_x)
    test_pairwise_distance = euclidean_distances(test_x, train_x)

    indexes = np.argpartition(pairwise_distance, k, axis=1)[:, :k]
    test_indexes = np.argpartition(test_pairwise_distance, k, axis=1)[:, :k]
    train_x = train_y[indexes].squeeze()
    test_x = train_y[test_indexes].squeeze()

    bound_weights = [(0.0, 1.0) for _ in range(train_x.shape[-1])]
    result = differential_evolution(weighted_mean_squared_error, bound_weights, (train_x, train_y), maxiter=500,
                                    tol=1e-7)
    print(result['message'], " number of iterations = ", result['nit'])
    # get the chosen weights
    weights = result["x"]
    filled_nan = predict(weights, test_x)
    return pd.Series([filled_nan, final_temp_nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset_date_modified("0.01")
    x_nan = x_nan[x_nan.id == 18]
    x = x[x.id == 18]
    x_size = [10, 20, 30, 40, 50]
    for size in x_size:
        filled_users = x_nan.groupby("id").apply(fill_nan, size)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        print("x size = ", size)
        print(evaluate_dataframe(filled_users, mean_square_error))
        print()
