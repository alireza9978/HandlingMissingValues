import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.smart_star.load_dataset import get_dataset, root


def plot_random_users_usage(temp_df: pd.DataFrame):
    folders = ["hourly", "weekly", "monthly"]
    time_scales = ["1H", "1W", "1M"]
    temp_df = temp_df.set_index("date")

    def inner_user_plotter(inner_df: pd.DataFrame):
        user_id = inner_df.id.values[0]
        for time_scale, folder in zip(time_scales, folders):
            inner_df = inner_df.resample(time_scale).sum()[["usage"]]
            plt.plot(inner_df.usage)
            plt.savefig(root + f"plots/user_figures/user_usage_{folder}/{user_id}.jpeg")
            plt.close()

    temp_df.groupby("id").apply(inner_user_plotter)


def plot_users_mean_usage_in_day(temp_df: pd.DataFrame):
    temp_df["hour"] = temp_df.date.dt.hour
    mean_df = temp_df.groupby(["id", "hour"]).mean()
    mean_df["std"] = temp_df.groupby(["id", "hour"]).std()
    mean_df = mean_df.reset_index()
    x_axis = np.arange(1, 25)

    def inner_plot(inner_df: pd.DataFrame):
        user_id = inner_df['id'].values[0]
        name = "user = " + str(user_id)
        plt.errorbar(x_axis, inner_df["usage"], inner_df["std"], marker='^', label=name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(root + f"plots/user_figures/user_mean_single_day/{user_id}.jpeg")
        plt.close()

    mean_df.groupby(["id"]).apply(inner_plot)


if __name__ == '__main__':
    x, x_nan = get_dataset("0.01")
    # plot_users_mean_usage_in_day(x)
    plot_random_users_usage(x)
