import importlib

from src.measurements.Measurements import *

from src.methods.Simple import FirstObservationCarriedBackward, LastObservationCarriedForward, \
    Interpolation
from src.methods.Simple.TotalMean import TotalMean
from src.methods.Simple.TotalMedian import TotalMedian
from src.methods.MovingWindow import Mean, WeightedMean, ExponentialMean
from src.methods.ARIMA.ARIMA import Arima
from src.methods.SVR.SVR import Svr
from src.preprocessing.insert_nan import nan_percents_str
from src.preprocessing.load_dataset import get_dataset, get_dataset_fully_modified_date
from src.preprocessing.load_dataset import root as root_path
from src.utils.parallelizem import apply_parallel

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]
measures_name = ["mse", "mae", "mape"]

methods_trainable = [
    TotalMean,
    TotalMedian,
    Arima,
    Svr
]

methods_trainable_name = [
    TotalMean.get_name(),
    TotalMedian.get_name(),
    Arima.get_name(),
    Svr.get_name(),
]

methods_trainable_params = [
    TotalMean.get_train_params(),
    TotalMedian.get_train_params(),
    Arima.get_train_params(),
    Svr.get_train_params(),
]

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
method_single_feature_param = [
    Mean.fill_nan,
    WeightedMean.fill_nan,
    ExponentialMean.fill_nan,
]

method_name_single_feature_param = [
    Mean.get_name(),
    WeightedMean.get_name(),
    ExponentialMean.get_name(),
]

method_single_feature_param_value = [
    Mean.get_params(),
    WeightedMean.get_params(),
    ExponentialMean.get_params(),
]

methods_multiple_feature = [
    SVR.fill_nan
]

method_name_multiple_feature = [
    SVR.get_name(),
]


#
# methods_multiple_feature_multi_params = [
#     importlib.import_module("KNN.KNNImputer")]
#
# methods_name_multiple_feature_multi_params = [
#     "KNN Imputer"]


# methods_complete_feature = [importlib.import_module("Regression.EveryDayLinear"), ]
#
# method_name_complete_feature = ["Regression Every Day", ]


def fill_nan(temp_x: pd.DataFrame, temp_x_nan: pd.DataFrame, fill_nan_method, params=None) -> pd.DataFrame:
    temp_filled_users = apply_parallel(temp_x_nan.groupby("id"), fill_nan_method, params)
    temp_filled_users = temp_filled_users.join(temp_x[["usage"]])
    return temp_filled_users
