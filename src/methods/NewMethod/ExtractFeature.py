import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from src.preprocessing.load_dataset import get_dataset
from src.utils.Dataset import get_random_user, load_error
from src.utils.Methods import method_name_single_feature, method_name_single_feature_param, \
    method_single_feature_param_value
from src.utils.Methods import measures_name
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


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


def add_methods_result(temp_df: pd.DataFrame):
    method_columns = []
    for name in method_name_single_feature:
        error_df = load_error(nan_percent, name, measures_name[0])
        temp_columns = error_df.columns.to_list()
        temp_columns[-1] = name
        method_columns.append(name)
        error_df.columns = temp_columns
        temp_df = temp_df.join(error_df[[name, "index"]].set_index("index"))
    for name, params in zip(method_name_single_feature_param, method_single_feature_param_value):
        for param in params:
            error_df = load_error(nan_percent, name, measures_name[0], param)
            temp_columns = error_df.columns.to_list()
            temp_name = name + str(param)
            temp_columns[-1] = temp_name
            method_columns.append(temp_name)
            error_df.columns = temp_columns
            temp_df = temp_df.join(error_df[[temp_name, "index"]].set_index("index"))
    return temp_df, method_columns


def add_label(temp_df: pd.DataFrame, columns_name: list):
    temp_df["label"] = temp_df[columns_name].to_numpy().argmin(axis=1)
    temp_df["minimum_error"] = temp_df[columns_name].to_numpy().min(axis=1)
    temp_error_df = temp_df[columns_name]
    return temp_df, temp_error_df


def calculate_error(error_df, prediction):
    errors = error_df.to_numpy()
    total = 0
    for i in range(prediction.shape[0]):
        total += errors[i][prediction[i]]

    return total / prediction.shape[0]


def generate_train_test(feature_df, error_df):
    x_train = feature_df[train_x_columns]
    x_train = x_train.dropna()
    y_train = feature_df["label"][x_train.index]
    error_df = error_df.loc[x_train.index]
    x_train, x_test, y_train, y_test, error_df_train, error_df_test = train_test_split(x_train, y_train, error_df,
                                                                                       test_size=0.3)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, error_df_train, error_df_test


def classification(x_train, x_test, y_train):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    temp_train_prediction = clf.predict(x_train)
    temp_y_prediction = clf.predict(x_test)
    return temp_train_prediction, temp_y_prediction


# def classification_ann(x_train, x_test, y_train):
#     inputs = layers.Input(shape=(input_shape,))
#     model = layers.Dense(32, activation="relu")(inputs)
#     model = layers.Dropout(0.25)(model)
#     model = layers.Dense(64, activation="relu")(model)
#     model = layers.Dropout(0.25)(model)
#     model = layers.Dense(16, activation="relu")(model)
#     model = layers.Dropout(0.25)(model)
#     output = layers.Dense(1, activation="sigmoid")(model)
#     model = keras.Model(inputs, output)
#     model.compile("adam", "mean_squared_error")
#     return temp_train_prediction, temp_y_prediction


if __name__ == '__main__':
    nan_percent = "0.01"
    x, x_nan = get_dataset(nan_percent)
    (x, x_nan) = get_random_user(x, x_nan)
    moving_features = x_nan.groupby("id").apply(calculate_feature, 12)
    moving_features = moving_features.reset_index(level=0)
    train_x_columns = moving_features.columns.copy()
    moving_features, methods_name = add_methods_result(moving_features)
    moving_features, usage_error_df = add_label(moving_features, methods_name)
    method_results = usage_error_df.mean()

    train_x, test_x, train_y, test_y, train_error_df, test_error_df = generate_train_test(moving_features,
                                                                                          usage_error_df)
    train_prediction, y_prediction = classification(train_x, test_x, train_y)

    print("best mse: ", moving_features["minimum_error"].mean())
    print("best mse for single method: ", method_results[method_results.argmin()],
          method_results.index[method_results.argmin()])
    print("train: ")
    print("accuracy: ", accuracy_score(train_y, train_prediction))
    print("mse: ", calculate_error(train_error_df, train_prediction))
    print("test: ")
    print("accuracy: ", accuracy_score(test_y, y_prediction))
    print("mse: ", calculate_error(test_error_df, y_prediction))
