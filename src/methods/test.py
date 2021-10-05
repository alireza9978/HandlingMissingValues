import importlib

from src.measurements.Measurements import mean_square_error
from src.preprocessing.load_dataset import get_dataset

methods = [importlib.import_module("Simple.Total Mean"),
           importlib.import_module("Simple.First Observation Carried Backward"),
           importlib.import_module("Simple.Last Observation Carried Forward")]

method_name = ["Total Mean",
               "First Observation Carried Backward",
               "Last Observation Carried Forward"]

x, x_nan = get_dataset()
for i in range(len(method_name)):
    print(method_name[i], mean_square_error(x, methods[i].fill_nan(x_nan)))
