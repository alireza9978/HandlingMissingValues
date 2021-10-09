import importlib

from src.measurements.Measurements import *
from src.preprocessing.load_dataset import get_dataset

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]

methods = [importlib.import_module("Simple.Total Mean"),
           importlib.import_module("Simple.Total Median"),
           importlib.import_module("Simple.First Observation Carried Backward"),
           importlib.import_module("Simple.Last Observation Carried Forward"),
           importlib.import_module("Simple.Interpolation")]

method_name = ["Total Mean",
               "Total Median",
               "First Observation Carried Backward",
               "Last Observation Carried Forward",
               "Interpolation"]

x, x_nan = get_dataset()
for i in range(len(method_name)):
    filled_users = x_nan.groupby("id").apply(methods[i].fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(method_name[i])
    for measure in measures:
        print(evaluate_dataframe(filled_users, measure))
    print()
