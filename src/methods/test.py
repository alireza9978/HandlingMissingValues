import importlib

from src.measurements.Measurements import calculate_measures
from src.preprocessing.load_dataset import get_dataset

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
    filled_x, nan_index = methods[i].fill_nan(x_nan)
    print(method_name[i])
    calculate_measures(x[nan_index], filled_x)
    print()
