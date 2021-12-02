from src.measurements.Measurements import *
from src.methods.ARIMA.ARIMA import Arima
from src.methods.MovingWindow.ExponentialMean import ExponentialMean
from src.methods.MovingWindow.MovingMean import MovingMean
from src.methods.MovingWindow.WeightedMean import WeightedMean
from src.methods.Simple.Interpolation import Interpolation
from src.methods.Simple.FirstObservationCarriedBackward import FirstObservationCarriedBackward
from src.methods.Simple.LastObservationCarriedForward import LastObservationCarriedForward
from src.methods.SVR.SVR import Svr
from src.methods.Simple.TotalMean import TotalMean
from src.methods.Simple.TotalMedian import TotalMedian
from src.utils.parallelizem import apply_parallel

measures = [mean_square_error, mean_absolute_error, mean_absolute_percentage_error]
measures_name = ["mse", "mae", "mape"]

methods_trainable = [
    TotalMean,
    TotalMedian,
    Interpolation,
    FirstObservationCarriedBackward,
    LastObservationCarriedForward,
    WeightedMean,
    MovingMean,
    ExponentialMean,
    Arima,
    Svr,
]


def fill_nan(temp_x: pd.DataFrame, temp_x_nan: pd.DataFrame, fill_nan_method, params=None) -> pd.DataFrame:
    temp_filled_users = apply_parallel(temp_x_nan.groupby("id"), fill_nan_method, params)
    temp_filled_users = temp_filled_users.join(temp_x[["usage"]])
    return temp_filled_users
