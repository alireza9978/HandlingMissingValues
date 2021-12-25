import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.insert_nan import nan_percents, nan_percents_str
from src.preprocessing.load_dataset import root
from src.utils.parallelizem import apply_parallel
from visualisation.ErrorPloting import clean_result_df

method_name_single_feature_window = [
    "Moving Window Mean",
    "Moving Window Weighted Mean",
    "Moving Window Exponential Mean",
]


def plot_result(temp_df: pd.DataFrame):
    x_axis_count = len(nan_percents)
    x = np.arange(x_axis_count)  # the label locations
    plt.figure(figsize=(10, 5))
    temp_df["nan_percent"] = temp_df["nan_percent"].astype(float)

    def plot_single_method(inner_df: pd.DataFrame):
        a = inner_df["nan_percent"].to_list()
        for b in nan_percents:
            if not a.__contains__(b):
                temp_row = inner_df.iloc[0]
                temp_row["nan_percent"] = b
                temp_row["mse"] = np.nan
                temp_row["mae"] = np.nan
                temp_row["mape"] = np.nan
                inner_df = inner_df.append(temp_row)

        y = inner_df.sort_values("nan_percent")["mse"].to_numpy()
        label = inner_df["model"].values[0]
        y[y > 2] = np.nan
        mask = ~np.isnan(y)
        temp_x = x[mask]
        temp_y = y[mask]
        plt.plot(temp_x, temp_y, label=label, )

    temp_df.groupby("model").apply(plot_single_method)

    plt.ylabel("Mean Squared Error")
    plt.xlabel("Nan Percent")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(x, nan_percents_str)
    plt.tight_layout()
    plt.show()


def load_results():
    root_path = root + "results/models/"
    merged_df = pd.DataFrame()
    for item in os.listdir(root_path):
        if item.endswith(".csv"):
            item_path = root_path + item
            temp_read_df = pd.read_csv(item_path)
            merged_df = merged_df.append(temp_read_df)

    merged_df = merged_df.dropna()
    return merged_df


def merge_windows(temp_df: pd.DataFrame):
    def inner_selection(inner_df: pd.DataFrame):
        temp_value = inner_df.iloc[np.argmin(inner_df["Mean Square Error"])]
        return temp_value

    for name in method_name_single_feature_window:
        temp_df["temp_col"] = temp_df["Method"].apply(lambda x: x.startswith(name))
        selected_df = apply_parallel(temp_df[temp_df["temp_col"]].groupby("Nan Percent"), inner_selection)
        selected_df["Method"] = name
        temp_df = temp_df[~temp_df["temp_col"]].append(selected_df)
        temp_df = temp_df.drop(columns=["temp_col"])

    return temp_df


def plot_moving_windows(temp_df: pd.DataFrame):
    x_axis_count = len(temp_df["Nan Percent"].unique())
    x = np.arange(x_axis_count)  # the label locations
    plt.figure(figsize=(10, 5))

    def inner_values(inner_df: pd.DataFrame):
        return pd.Series([inner_df["Mean Square Error"].mean(), inner_df["Mean Square Error"].std()])

    for name in method_name_single_feature_window:
        temp_df["temp_col"] = temp_df["Method"].apply(lambda temp_value: temp_value.startswith(name))
        temp_method_df = temp_df[temp_df.temp_col]
        result = temp_method_df.groupby(["Nan Percent"]).apply(inner_values)
        temp_df = temp_df[~temp_df["temp_col"]]
        temp_df = temp_df.drop(columns=["temp_col"])

        temp_index = result.index.to_list()
        for i in nan_percents:
            if not temp_index.__contains__(i):
                result = result.append(pd.Series([np.nan, np.nan]), ignore_index=True)

        plt.errorbar(x, result[0], result[1], marker='^', label=name)

    plt.ylabel("Mean Squared Error")
    plt.xlabel("Nan Percent")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(x, nan_percents_str)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = clean_result_df()
    # plot_moving_windows(df)
    # df = merge_windows(df)
    plot_result(df)
