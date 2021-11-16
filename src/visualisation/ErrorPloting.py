import matplotlib.pyplot as plt

from src.measurements.Measurements import *
from src.methods.MovingWindow import Mean
from src.preprocessing.load_dataset import get_dataset
from src.preprocessing.load_dataset import root
from src.utils.parallelizem import apply_parallel


def plot_highest_errors(main_df, filled_users, model_name):
    def inner_plot(temp_df, k):
        temp_user_id = temp_df[2].id.values[0]
        difference_to_real = (temp_df[0] - temp_df[2].usage.to_numpy().reshape(-1, 1))
        indexes = np.argpartition(difference_to_real.squeeze(), -k)[-k:]
        fig, axs = plt.subplots(k, sharex="all", figsize=(8, 12))
        fig.suptitle(f'user {temp_user_id}\n highest error in prediction with model {model_name}')

        for i, temp_index in enumerate(indexes):
            main_df_index = temp_df[1][temp_index]
            real_usage = main_df.loc[main_df_index - 12:main_df_index + 12].usage.to_numpy()
            predicted_usage = temp_df[0][temp_index]
            axs[i].plot(np.arange(25), real_usage)
            axs[i].plot([12], predicted_usage, marker="x")

        for ax in axs.flat:
            ax.set(xlabel='time', ylabel='usage')
        fig.tight_layout()
        fig.savefig(root + f"results/errors/{model_name}_{temp_user_id}.jpeg")
        fig.close()

    filled_users.apply(inner_plot, args=[10], axis=1)


def run_model(main_df, nan_df, model, params):
    filled_users = apply_parallel(nan_df.groupby("id"), model, params)
    filled_users[2] = filled_users[1].apply(lambda idx: main_df.loc[idx])
    return filled_users


if __name__ == '__main__':
    x, x_nan = get_dataset("0.01")
    window_sizes = [4, 6, 8, 10, 12, 24, 48, 168, 720]
    for size in window_sizes:
        result = run_model(x, x_nan, Mean.fill_nan, size)
        plot_highest_errors(x, result, "moving_window_mean")
