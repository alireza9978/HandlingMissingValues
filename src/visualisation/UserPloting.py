import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.load_dataset import get_dataset, root
import numpy as np


def plot_random_users_usage(temp_df: pd.DataFrame):
    week_x = temp_df.set_index("date")
    week_x = week_x.groupby("id").apply(lambda df: df.resample("1W").sum()[["usage"]]).reset_index()
    users_id = week_x.id.unique()
    sample_users = np.random.choice(users_id, 20)
    for user_id in sample_users:
        temp_user = week_x[week_x.id == user_id]
        plt.plot(temp_user.usage)
        plt.savefig(root + f"results/user_figures/{user_id}.jpeg")
        plt.close()


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
        plt.savefig(root + f"results/user_figures/{user_id}.jpeg")
        plt.close()

    mean_df.groupby(["id"]).apply(inner_plot)


if __name__ == '__main__':
    x, x_nan = get_dataset("0.01")
    plot_users_mean_usage_in_day(x)
