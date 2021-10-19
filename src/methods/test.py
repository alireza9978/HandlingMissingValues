import importlib

from src.measurements.Measurements import *
from src.preprocessing.load_dataset import get_dataset, get_dataset_fully_modified_date
from src.preprocessing.load_dataset import root as root_path
from src.utils.parallelizem import apply_parallel
from src.visualisation.visualization import plot_result

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]

methods_single_feature = [
    importlib.import_module("Simple.Total Mean"),
    importlib.import_module("Simple.Total Median"),
    importlib.import_module("Simple.FirstObservationCarriedBackward"),
    importlib.import_module("Simple.LastObservationCarriedForward"),
    importlib.import_module("Simple.Interpolation")
]

method_name_single_feature = [
    "Total Mean",
    "Total Median",
    "First Observation Carried Backward",
    "Last Observation Carried Forward",
    "Interpolation"
]

methods_multiple_feature = [importlib.import_module("Regression.Linear"),
                            importlib.import_module("Hot Deck.Hot Deck")]

method_name_multiple_feature = ["Linear Regression",
                                "Hot Deck"]

result_df = pd.DataFrame()
x, x_nan = get_dataset()
for i in range(len(method_name_single_feature)):
    filled_users = x_nan.groupby("id").apply(methods_single_feature[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(method_name_single_feature[i])
    for measure in measures:
        print(evaluate_dataframe(filled_users, measure))

x, x_nan = get_dataset_fully_modified_date()
for i in range(len(methods_multiple_feature)):
    filled_users = apply_parallel(x_nan.groupby("id"), methods_multiple_feature[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(method_name_multiple_feature[i])
    temp_result_list = [method_name_single_feature[i]]
    for measure in measures:
        measured_value = evaluate_dataframe(filled_users, measure)
        temp_result_list.append(measured_value)
        print(measured_value)


result_df.columns = ["Method", "Mean Square Error", "Mean Absolute Error", "Mean Absolute Percentage Error"]
# plot_result(result_df)

result_df.to_csv(root_path + "results/methods result.csv", index=False)
