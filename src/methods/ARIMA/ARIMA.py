from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset
from src.utils.parallelizem import apply_parallel


def fill_nan(user_data):
    user = user_data.copy()
    # getting the nan indexes
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    model = ARIMA(user['usage'], order=(2, 1, 2))
    results = model.fit()
    user['usage'] = pd.Series(results.fittedvalues, copy=True)
    predictions = user['usage'][nan_index]
    # plt.plot(predictions)
    # plt.show()
    # plt.plot(user['usage'])
    # plt.show()
    return pd.Series([predictions, nan_index])


if __name__ == '__main__':
    x, x_nan = get_dataset(nan_percent='0.05')
    x_nan = x_nan[x_nan.id == 100]
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    # filled_users = x_nan.groupby("id").apply(fill_nan)
    # filled_users = fill_nan(x_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(



        evaluate_dataframe(filled_users, mean_square_error))
