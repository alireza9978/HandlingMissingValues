from builtins import enumerate

import matplotlib.pyplot as plt

from src.measurements.Measurements import *
from src.methods.ARIMA import ARIMA
from src.methods.MovingWindow import MovingMean
from src.methods.Simple import Interpolation
from src.preprocessing.load_dataset import get_dataset
from src.preprocessing.load_dataset import root
from src.utils.Dataset import get_random_user
from src.utils.parallelizem import apply_parallel


def plot_highest_errors(main_df, nan_df, filled_users, model_name):
    def inner_plot(temp_df, k):
        temp_user_id = temp_df[2].id.values[0]
        difference_to_real = (temp_df[0] - temp_df[2].usage.to_numpy().reshape(-1, 1))
        indexes = np.argpartition(difference_to_real.squeeze(), -k)[-k:]
        fig, axs = plt.subplots(k, sharex="all", figsize=(8, 12))
        fig.suptitle(f'user {temp_user_id}\n highest error in prediction with model {model_name}')
        max_index = main_df.index[-1]
        min_index = main_df.index[0]

        for i, temp_index in enumerate(indexes):
            main_df_index = temp_df[1][temp_index]
            real_usage = main_df.loc[main_df_index - 12:main_df_index + 12].usage.to_numpy()
            nan_usage = nan_df.loc[main_df_index - 12:main_df_index + 12].usage.to_numpy()
            right_padding = 0
            left_padding = 0
            if main_df_index + 12 > max_index:
                right_padding = 12 - (max_index - main_df_index)
            if main_df_index - 12 < min_index:
                left_padding = 12 - (main_df_index - min_index)
            real_usage = np.pad(real_usage, [left_padding, right_padding], mode="constant", constant_values=np.nan)
            nan_usage = np.pad(nan_usage, [left_padding, right_padding], mode="constant", constant_values=np.nan)
            nan_indexes = np.isnan(nan_usage)
            x_axis_indexes = np.arange(real_usage.shape[0])
            if nan_indexes.sum() > 1:
                print("here")
            predicted_usage = temp_df[0][temp_index]
            axs[i].plot(x_axis_indexes, real_usage)
            axs[i].plot(x_axis_indexes[nan_indexes], real_usage[nan_indexes], marker="o", linestyle='None')
            axs[i].plot([12], predicted_usage, marker="x")

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='usage')
        fig.tight_layout()
        fig.savefig(root + f"results/errors/{model_name}_{temp_user_id}.jpeg")
        plt.close()

    filled_users.apply(inner_plot, args=[10], axis=1)


def run_model(main_df, nan_df, model, temp_params=None):
    filled_users = apply_parallel(nan_df.groupby("id"), model, temp_params)
    filled_users[2] = filled_users[1].apply(lambda idx: main_df.loc[idx])
    return filled_users


if __name__ == '__main__':
    selected_methods = [Mean.fill_nan, ARIMA.fill_nan, Interpolation.fill_nan]
    selected_params = [[4, 6, 8, 10, 12, 24, 48, 168, 720], None, None]
    selected_methods_name = [Mean.get_name(), ARIMA.get_name(), Interpolation.get_name()]
    nan_percents = ["0.01", "0.1", "0.2"]

    for nan_percent in nan_percents:
        x, x_nan = get_dataset(nan_percent)
        x, x_nan = get_random_user(x, x_nan)
        for j, method in enumerate(selected_methods):
            params = selected_params[j]
            if params is not None:
                for param in params:
                    result = run_model(x, x_nan, method, param)
                    plot_highest_errors(x, x_nan, result, f"{selected_methods_name[j]}_param{param}_{nan_percent}")
            else:
                result = run_model(x, x_nan, method)
                plot_highest_errors(x, x_nan, result, f"{selected_methods_name[j]}_{nan_percent}")
        print(f"finish plotting in {nan_percent} percent")
