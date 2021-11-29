from src.preprocessing.load_dataset import get_dataset
from src.utils.Dataset import load_error

x, x_nan = get_dataset("0.1")
x_nan = x_nan[x_nan.id == 100]
nan_row = x_nan[x_nan["usage"].isna()]
indices = nan_row.index.to_numpy()
methods = ['arima', 'interpolation', 'last_observation_carried_forward',
           'total_mean']
for method in methods:
    error_df = load_error("0.1", method, 'mse')
    error_df = error_df.set_index("index").loc[indices]
    print('method name:', method)
    print('correlation: ', error_df.corr(method='pearson')['usage'][0])
