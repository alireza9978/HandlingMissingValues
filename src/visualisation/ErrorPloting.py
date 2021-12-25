from builtins import enumerate
from pathlib import Path

import matplotlib.pyplot as plt

from src.measurements.Measurements import *
from src.preprocessing.load_dataset import root
from src.utils.Dataset import load_all_methods_result
from src.utils.Methods import measures_name


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


def select_best_param(temp_df: pd.DataFrame):
    temp_result = temp_df[['mse', 'mae', 'mape', 'mse_test', 'mae_test', 'mape_test']].mean()
    temp_result["nan_percent"] = temp_df["nan_percent"].values[0]
    temp_result["model"] = temp_df["model"].values[0]
    return temp_result


def clean_result_df():
    train_df, test_df = load_all_methods_result()
    train_df = train_df.sort_values("mse")
    train_df = train_df[train_df.mse < 5]
    train_df = train_df.set_index(["nan_percent", "model", "params"])
    test_df = test_df.set_index(["nan_percent", "model", "params"])
    total_df = train_df.join(test_df, rsuffix="_test")
    total_df = total_df.reset_index()
    total_df = total_df.groupby(["nan_percent", "model"]).apply(select_best_param)
    total_df = total_df.reset_index(drop=True)
    return total_df


def plot_results_df(results: pd.DataFrame):
    def plot_single_nan_percents_result(inner_df: pd.DataFrame):
        nan_percent = inner_df.nan_percent.values[0]
        temp_path = root + f"plots/methods/{nan_percent}"
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        x = np.arange(inner_df.shape[0])
        for measure in measures_name:
            selected_columns = [measure, "model"]
            selected_df = inner_df[selected_columns].sort_values("model")
            plt.figure(figsize=(5, 7))
            plt.bar(x, selected_df[measure])
            plt.ylabel(measure)
            plt.xlabel("Method Name")
            plt.xticks(x, selected_df["model"])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(root + f"plots/methods/{nan_percent}/train_{measure}.jpeg")
            plt.close()

            selected_columns = [measure + "_test", "model"]
            selected_df = inner_df[selected_columns].sort_values("model")
            plt.figure(figsize=(5, 7))
            plt.bar(x, selected_df[selected_columns[0]])
            plt.ylabel(measure)
            plt.xlabel("Method Name")
            plt.xticks(x, selected_df["model"])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(root + f"plots/methods/{nan_percent}/test_{measure}.jpeg")
            plt.close()

    results.groupby("nan_percent").apply(plot_single_nan_percents_result)


if __name__ == '__main__':
    clean_df = clean_result_df()
    plot_results_df(clean_df)
