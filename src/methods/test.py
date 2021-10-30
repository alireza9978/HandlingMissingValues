import importlib

from src.measurements.Measurements import *
from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_dataset, get_dataset_fully_modified_date, get_complete_dataset
from src.preprocessing.load_dataset import root as root_path
from src.utils.parallelizem import apply_parallel

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]

methods_single_feature = [
    importlib.import_module("Simple.Total Mean"),
    importlib.import_module("Simple.Total Median"),
    importlib.import_module("Simple.FirstObservationCarriedBackward"),
    importlib.import_module("Simple.LastObservationCarriedForward"),
    importlib.import_module("Simple.Interpolation"),
    importlib.import_module("ARIMA.ARIMA"),
]

method_name_single_feature = [
    "Total Mean",
    "Total Median",
    "First Observation Carried Backward",
    "Last Observation Carried Forward",
    "Interpolation",
    "ARIMA"
]

method_single_feature_window = [
    importlib.import_module("MovingWindow.Mean"),
    importlib.import_module("MovingWindow.WeightedMean"),
    importlib.import_module("MovingWindow.ExponentialMean")
]

method_name_single_feature_window = [
    "Moving Window Mean",
    "Moving Window Weighted Mean",
    "Moving Window Exponential Mean",
]

methods_multiple_feature = [importlib.import_module("Regression.Linear"),
                            importlib.import_module("Hot Deck.Hot Deck"),
                            # importlib.import_module("Jung.MultiLayerPerceptron"),
                            importlib.import_module("KNN.KNNImputer"),
                            importlib.import_module("SVR.SVR")]

method_name_multiple_feature = ["Linear Regression",
                                "Hot Deck",
                                # "Multi Layer Perceptron",
                                "KNN Imputer",
                                "SVR"]

methods_complete_feature = [importlib.import_module("Regression.EveryDayLinear"), ]

method_name_complete_feature = ["Regression Every Day", ]

for nan_percent in nan_percents_str[0:1]:
    result_df = pd.DataFrame()
    x, x_nan = get_dataset(nan_percent)
    for i in range(len(method_name_single_feature)):
        filled_users = apply_parallel(x_nan.groupby("id"), methods_single_feature[i].fill_nan)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        temp_result_list = [method_name_single_feature[i], nan_percent]
        for measure in measures:
            measured_value = evaluate_dataframe(filled_users, measure)
            temp_result_list.append(measured_value)
        print("method {} finished".format(method_name_single_feature[i]))
        result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)

    for i in range(len(method_name_single_feature_window)):
        window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]
        for window_index in range(len(window_sizes)):
            window_size = window_sizes[window_index]
            filled_users = apply_parallel(x_nan.groupby("id"), method_single_feature_window[i].fill_nan, window_size)
            filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
            temp_result_list = ["{}_window_{}".format(method_name_single_feature_window[i], window_size), nan_percent]
            for measure in measures:
                measured_value = evaluate_dataframe(filled_users, measure)
                temp_result_list.append(measured_value)
            result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)
        print("method {} finished".format(method_name_single_feature_window[i]))

    x, x_nan = get_dataset_fully_modified_date(nan_percent)
    for i in range(len(methods_multiple_feature)):
        filled_users = apply_parallel(x_nan.groupby("id"), methods_multiple_feature[i].fill_nan)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        temp_result_list = [method_name_multiple_feature[i], nan_percent]
        for measure in measures:
            measured_value = evaluate_dataframe(filled_users, measure)
            temp_result_list.append(measured_value)
        result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)
        print("method {} finished".format(method_name_multiple_feature[i]))

    x, x_nan = get_complete_dataset(nan_percent)
    for i in range(len(methods_complete_feature)):
        filled_users = apply_parallel(x_nan.groupby("id"), methods_complete_feature[i].fill_nan)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        temp_result_list = [method_name_complete_feature[i]]
        for measure in measures:
            measured_value = evaluate_dataframe(filled_users, measure)
            temp_result_list.append(measured_value)
        print("method {} finished".format(method_name_complete_feature[i]))
        result_df = result_df.append(pd.Series(temp_result_list), ignore_index=True)

    result_df.columns = ["Method", "Nan Percent", "Mean Square Error", "Mean Absolute Error",
                         "Mean Absolute Percentage Error"]
    # plot_result(result_df)

    result_df.to_csv(root_path + f"results/methods result_{nan_percent}.csv", index=False)
