import importlib

from src.measurements.Measurements import *

from src.methods.Simple import FirstObservationCarriedBackward, TotalMedian, LastObservationCarriedForward, \
    Interpolation, TotalMean
from src.methods.ARIMA import ARIMA
from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_dataset, get_dataset_fully_modified_date
from src.preprocessing.load_dataset import root as root_path
from src.utils.parallelizem import apply_parallel

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]
measures_name = ["mse", "mae", "mape"]

methods_single_feature = [
    TotalMean.fill_nan,
    TotalMedian.fill_nan,
    FirstObservationCarriedBackward.fill_nan,
    LastObservationCarriedForward.fill_nan,
    Interpolation.fill_nan,
    ARIMA.fill_nan
]

method_name_single_feature = [
    TotalMean.get_name(),
    TotalMedian.get_name(),
    FirstObservationCarriedBackward.get_name(),
    LastObservationCarriedForward.get_name(),
    Interpolation.get_name(),
    ARIMA.get_name()
]


#
# method_single_feature_window = [
#     importlib.import_module("MovingWindow.Mean"),
#     importlib.import_module("MovingWindow.WeightedMean"),
#     importlib.import_module("MovingWindow.ExponentialMean")
# ]
#
# method_name_single_feature_window = [
#     "Moving Window Mean",
#     "Moving Window Weighted Mean",
#     "Moving Window Exponential Mean",
# ]
#
# methods_multiple_feature = [importlib.import_module("Regression.Linear"),
#                             # importlib.import_module("Hot Deck.Hot Deck"),
#                             # importlib.import_module("Jung.MultiLayerPerceptron"),
#                             importlib.import_module("SVR.SVR")]
#
# method_name_multiple_feature = ["Linear Regression",
#                                 # "Hot Deck",
#                                 # "Multi Layer Perceptron",
#                                 "SVR"]
#
# methods_multiple_feature_multi_params = [
#     importlib.import_module("KNN.KNNImputer")]
#
# methods_name_multiple_feature_multi_params = [
#     "KNN Imputer"]


# methods_complete_feature = [importlib.import_module("Regression.EveryDayLinear"), ]
#
# method_name_complete_feature = ["Regression Every Day", ]


def fill_nan(temp_x: pd.DataFrame, temp_x_nan: pd.DataFrame, fill_nan_method) -> pd.DataFrame:
    temp_filled_users = apply_parallel(temp_x_nan.groupby("id"), fill_nan_method)
    temp_filled_users = temp_filled_users.reset_index(level=0)
    temp_filled_users = temp_filled_users.join(temp_x.drop(columns=["id", "date"]))
    return temp_filled_users
