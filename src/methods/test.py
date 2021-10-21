import importlib

from src.measurements.Measurements import *
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
]

method_name_single_feature = [
    "Total Mean",
    "Total Median",
    "First Observation Carried Backward",
    "Last Observation Carried Forward",
    "Interpolation",
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
                            importlib.import_module("Hot Deck.Hot Deck")]

method_name_multiple_feature = ["Linear Regression",
                                "Hot Deck"]

methods_complete_feature = [importlib.import_module("Regression.EveryDayLinear"),
                            ]

method_name_complete_feature = ["Regression on Single Day",
                                ]

result_df = pd.DataFrame()
x, x_nan = get_dataset()
for i in range(len(method_name_single_feature)):
    filled_users = x_nan.groupby("id").apply(methods_single_feature[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    temp_result_list = [method_name_single_feature[i]]
    for measure in measures:
        measured_value = evaluate_dataframe(filled_users, measure)
        temp_result_list.append(measured_value)
    print("method {} finished".format(method_name_single_feature[i]))
    result_df.append(pd.Series(temp_result_list), ignore_index=True)

for i in range(len(method_name_single_feature_window)):
    window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]
    for window_index in range(len(window_sizes)):
        window_size = window_sizes[window_index]
        method_single_feature_window[i].window_size = window_size
        filled_users = x_nan.groupby("id").apply(method_single_feature_window[i].fill_nan)
        filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
        temp_result_list = ["{}_window_{}".format(method_name_single_feature_window[i], window_size)]
        for measure in measures:
            measured_value = evaluate_dataframe(filled_users, measure)
            temp_result_list.append(measured_value)
        result_df.append(pd.Series(temp_result_list), ignore_index=True)
    print("method {} finished".format(method_name_single_feature_window[i]))

x, x_nan = get_dataset_fully_modified_date()
for i in range(len(methods_multiple_feature)):
    filled_users = apply_parallel(x_nan.groupby("id"), methods_multiple_feature[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    temp_result_list = [methods_multiple_feature[i]]
    for measure in measures:
        measured_value = evaluate_dataframe(filled_users, measure)
        temp_result_list.append(measured_value)
    print("method {} finished".format(method_name_multiple_feature[i]))
    result_df.append(pd.Series(temp_result_list), ignore_index=True)

x, x_nan = get_complete_dataset()
for i in range(len(methods_complete_feature)):
    filled_users = x_nan.groupby("id").apply(methods_complete_feature[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    temp_result_list = [method_name_complete_feature[i]]
    for measure in measures:
        measured_value = evaluate_dataframe(filled_users, measure)
        temp_result_list.append(measured_value)
    print("method {} finished".format(method_name_complete_feature[i]))
    result_df.append(pd.Series(temp_result_list), ignore_index=True)

result_df.columns = ["Method", "Mean Square Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]
# plot_result(result_df)

result_df.to_csv(root_path + "results/methods result.csv", index=False)
