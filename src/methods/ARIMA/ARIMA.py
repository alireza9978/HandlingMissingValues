import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.measurements.Measurements import evaluate_dataframe, mean_square_error, evaluate_dataframe_two
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel
import warnings


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    return scaler, user


def fill_nan(user_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user = user_data.copy()
        # getting the nan indexes
        nan_row = user[user["usage"].isna()]
        nan_index = nan_row.index.to_numpy()
        scaler, user = normalize_user_usage(user)
        model = ARIMA(user['usage'], order=(2, 1, 2))
        # results = model.fit(method_kwargs={"warn_convergence": False, "warn_value": False})
        results = model.fit()
        user['usage'] = pd.Series(results.fittedvalues, copy=True)
        predictions = scaler.inverse_transform(user['usage'][nan_index].to_numpy().reshape(-1, 1))
        return pd.DataFrame({"predicted_usage": predictions.squeeze()}, index=nan_index.squeeze())


if __name__ == '__main__':
    from src.utils.Methods import fill_nan as fn

    nan_percent = "0.01"
    x, x_nan = get_dataset(nan_percent)
    filled_users = fn(x, x_nan, fill_nan)
    error, error_df = evaluate_dataframe_two(filled_users, mean_square_error)
    print(error)
    print(error_df)


def get_name():
    return "arima"
