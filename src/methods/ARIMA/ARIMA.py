import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    return scaler, user


def fill_nan(user_data):
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
    return pd.Series([predictions, nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset(nan_percent='0.1')
    x_nan = x_nan[x_nan.id == 100]
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))


def get_name():
    return "ARIMA"
