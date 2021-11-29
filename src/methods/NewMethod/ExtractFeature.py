import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.preprocessing.load_dataset import get_dataset
from src.utils.Dataset import get_random_user, load_error, get_all_error_dfs, get_user_by_id
from src.utils.Methods import method_name_single_feature, method_name_single_feature_param, \
    method_single_feature_param_value
from src.utils.Methods import measures_name
from src.preprocessing.load_dataset import root


def calculate_feature(temp_df: pd.DataFrame, window_size: int):
    import swifter
    _ = swifter.config

    minimum_index = temp_df.index[0]
    maximum_index = temp_df.index[-1]

    nan_indexes = temp_df.usage.isna()
    result_df = pd.DataFrame()
    for temp_index in nan_indexes[nan_indexes].index:
        temp_minimum_index = max((temp_index - window_size, minimum_index))
        temp_maximum_index = min((temp_index + window_size, maximum_index))
        left_feature = temp_df.loc[temp_minimum_index:temp_index - 1].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        left_feature.index += "_left"
        right_feature = temp_df.loc[temp_index + 1:temp_maximum_index].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        right_feature.index += "_right"
        feature_row = pd.concat([left_feature, right_feature])
        feature_row.name = temp_index
        result_df = result_df.append(feature_row)

    return result_df


def add_methods_result(nan_percent):
    error_df = get_all_error_dfs(nan_percent, measures_name[0])
    method_columns = error_df.columns
    return error_df, method_columns


def add_label(temp_df: pd.DataFrame):
    label = temp_df.to_numpy().argmin(axis=1)
    minimum_error = temp_df.to_numpy().min(axis=1)
    temp_df['label'] = label
    temp_df['minimum_error'] = minimum_error
    return temp_df


def calculate_error(error_df, prediction):
    errors = error_df.to_numpy()
    total = 0
    for i in range(prediction.shape[0]):
        total += errors[i][prediction[i]]

    return total / prediction.shape[0]


def clustering_aggregation(n_clusters, train_x, train_y, test_x, test_y, train_error_df, test_error_df):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(train_x)
    temp = pd.DataFrame({'label': train_y, 'cluster': kmeans.labels_})

    # vote_results = temp.groupby('cluster').agg(pd.Series.mode)
    def city(temp_row):
        un = np.unique(temp_row.label, return_counts=True)
        return un[0][np.argmax(un[1])]

    vote_results = temp.groupby('cluster').apply(city)
    # vote_results['label'] = [i[0] if type(i) == np.ndarray else i for i in vote_results.label.to_list()]
    predictions = kmeans.predict(test_x)
    predictions = [vote_results.iloc[i] for i in predictions]
    errors = [test_error_df.iloc[i, predictions[i]] for i in range(len(predictions))]
    errors = np.array(errors)
    mse = errors.mean()
    return mse


if __name__ == '__main__':
    nan_percent = "0.1"
    x, x_nan = get_dataset(nan_percent)
    # (x, x_nan) = get_random_user(x, x_nan)
    results = []
    for id in range(1, 114):
        (t, t_nan) = get_user_by_id(x, x_nan,id)
        moving_features = t_nan.groupby("id").apply(calculate_feature, 12)
        moving_features = moving_features.reset_index(level=0).drop(columns=["id"])
        usage_error_df, methods_name = add_methods_result(nan_percent)
        usage_error_df = add_label(usage_error_df)
        method_results = usage_error_df.mean()

        train_x = moving_features
        train_x = train_x.dropna()
        train_y = usage_error_df["label"][train_x.index]
        usage_error_df = usage_error_df.loc[train_x.index]
        train_x, test_x, train_y, test_y, train_error_df, test_error_df = train_test_split(train_x, train_y,
                                                                                           usage_error_df,
                                                                                           test_size=0.3)

        for i in range(3, 10):
            result = clustering_aggregation(i, train_x, train_y, test_x, test_y, train_error_df, test_error_df)
            print('MSE in n_clusters {} is equal to {}'.format(i, result))
            results.append([id, i, result])
    results = pd.DataFrame(results, columns=["id", "n_clusters", "mse"])
    results.to_csv(Path(root + 'results/clustering_results_{}.csv'.format(nan_percent)), index=False)
        # clf = RandomForestClassifier(n_estimators=32, max_depth=5, min_samples_leaf=4)
        # clf.fit(train_x, train_y)
        # train_prediction = clf.predict(train_x)
        # y_prediction = clf.predict(test_x)
        #
        # print("best mse: ", moving_features["minimum_error"].mean())
        # print("best mse for single method: ", method_results[method_results.argmin()],
        #       method_results.index[method_results.argmin()])
        # print("train: ")
        # print("accuracy: ", accuracy_score(train_y, train_prediction))
        # print("mse: ", calculate_error(train_error_df, train_prediction))
        # print("test: ")
        # print("accuracy: ", accuracy_score(test_y, y_prediction))
        # print("mse: ", calculate_error(test_error_df, y_prediction))
