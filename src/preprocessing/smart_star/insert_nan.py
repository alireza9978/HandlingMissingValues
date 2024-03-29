from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.smart_star.load_dataset import root

nan_percents = [0.01, 0.05, 0.10, 0.15, 0.20, ]
# 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
nan_percents_str = ["0.01", "0.05", "0.1", "0.15", "0.2", ]


# "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]


def insert_nan(file_name: str):
    for percent in nan_percents:
        main_df = pd.read_csv(Path(root + "datasets/smart_star/{}.csv".format(file_name)))
        record_count = main_df.shape[0]
        random_index = np.random.choice(range(record_count), int(record_count * percent), replace=False)
        main_df.loc[random_index, "usage"] = np.nan
        print("nan percent = ", percent)
        print(main_df.isna().sum())
        main_df.to_csv(Path(root + "datasets/smart_star/with_nan/{}_{}.csv".format(file_name, percent)), index=False)


def insert_nan_train_test(file_name: str, train_percent: float, train_files: bool = True):
    if train_files:
        file_mode = "train"
    else:
        file_mode = "test"

    for percent in nan_percents:
        main_df = pd.read_csv(
            Path(root + "datasets/smart_star/train_test/{}_{}_{}.csv".format(file_name, file_mode, train_percent)))
        record_count = main_df.shape[0]
        random_index = np.random.choice(range(record_count), int(record_count * percent), replace=False)
        main_df.loc[random_index, "usage"] = np.nan
        print("nan percent = ", percent)
        print(main_df.isna().sum())
        main_df.to_csv(Path(
            root + "datasets/smart_star/train_test/with_nan/{}_{}_{}_nan_{}.csv".format(file_name, file_mode,
                                                                                        train_percent,
                                                                                        percent)), index=False)


def insert_nan_train_test_related(file_name: str, source_file_name: str, train_percent: float,
                                  train_files: bool = True):
    if train_files:
        file_mode = "train"
    else:
        file_mode = "test"

    for percent in nan_percents:
        main_df = pd.read_csv(
            Path(root + "datasets/train_test/{}_{}_{}.csv".format(file_name, file_mode, train_percent)))
        source_main_df = pd.read_csv(
            Path(root + "datasets/train_test/with_nan/{}_{}_{}_nan_{}.csv".format(source_file_name, file_mode,
                                                                                  train_percent,
                                                                                  percent)))
        main_df.loc[source_main_df[source_main_df.usage.isna()].index, "usage"] = np.nan
        print("nan percent = ", percent)
        print(main_df.isna().sum())
        main_df.to_csv(Path(
            root + "datasets/train_test/with_nan/{}_{}_{}_nan_{}.csv".format(file_name, file_mode, train_percent,
                                                                             percent)), index=False)


if __name__ == '__main__':
    file = "smart_star_hourly_final_with_date"
    source_file = "smart_star_hourly_final"
    train = 0.3
    insert_nan(file)
    # insert_nan_train_test(source_file, train)
    # insert_nan_train_test(source_file, train, False)
    # insert_nan_train_test_related(file, source_file, train)
    # insert_nan_train_test_related(file, source_file, train, False)
