from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.smart_star.convert_timestamps import convert_date

# root = "E:/HandlingMissingValues/"

root = "/home/alireza/projects/python/HandlingMissingValues/"


# root = "/home/ippbx/PycharmProjects/HandlingMissingValues/"
# root = 'h:/Projects/Datasets/Smart/'


def get_train_test_fully_modified_date(nan_percent: str, test_percent: float):
    file_name = "smart_star_hourly_fully_modified"
    train_df = pd.read_csv(Path(root + "datasets/train_test/{}_train_{}.csv".format(file_name, test_percent)))
    test_df = pd.read_csv(Path(root + "datasets/train_test/{}_test_{}.csv".format(file_name, test_percent)))
    train_nan_df = pd.read_csv(
        Path(root + "datasets/train_test/with_nan/{}_train_{}_nan_{}.csv".format(file_name, test_percent, nan_percent)))
    test_nan_df = pd.read_csv(
        Path(root + "datasets/train_test/with_nan/{}_test_{}_nan_{}.csv".format(file_name, test_percent, nan_percent)))
    return [((train_df, test_df, train_nan_df, test_nan_df), None)]


def get_train_test_dataset_triple(nan_percent: str):
    test_percent = "0.33"
    file_name = "smart_star_hourly"
    results = []
    for ch in ["a", "b", "c"]:
        train_df = pd.read_csv(
            Path(root + "datasets/train_test/{}_train_{}_{}.csv".format(file_name, test_percent, ch)))
        test_df = pd.read_csv(Path(root + "datasets/train_test/{}_test_{}_{}.csv".format(file_name, test_percent, ch)))
        train_nan_df = pd.read_csv(
            Path(root + "datasets/train_test/with_nan/{}_train_{}_{}_nan_{}.csv".format(file_name, test_percent, ch,
                                                                                        nan_percent)))
        test_nan_df = pd.read_csv(
            Path(root + "datasets/train_test/with_nan/{}_test_{}_{}_nan_{}.csv".format(file_name, test_percent, ch,
                                                                                       nan_percent)))
        train_df.date = pd.to_datetime(train_df.date)
        test_df.date = pd.to_datetime(test_df.date)
        train_nan_df.date = pd.to_datetime(train_nan_df.date)
        test_nan_df.date = pd.to_datetime(test_nan_df.date)
        results.append(((train_df, test_df, train_nan_df, test_nan_df), ch))
    return results


def get_train_test_dataset(nan_percent: str, test_percent: float):
    file_name = "smart_star_hourly"
    train_df = pd.read_csv(Path(root + "datasets/train_test/{}_train_{}.csv".format(file_name, test_percent)))
    test_df = pd.read_csv(Path(root + "datasets/train_test/{}_test_{}.csv".format(file_name, test_percent)))
    train_nan_df = pd.read_csv(
        Path(root + "datasets/train_test/with_nan/{}_train_{}_nan_{}.csv".format(file_name, test_percent, nan_percent)))
    test_nan_df = pd.read_csv(
        Path(root + "datasets/train_test/with_nan/{}_test_{}_nan_{}.csv".format(file_name, test_percent, nan_percent)))
    train_df.date = pd.to_datetime(train_df.date)
    test_df.date = pd.to_datetime(test_df.date)
    train_nan_df.date = pd.to_datetime(train_nan_df.date)
    test_nan_df.date = pd.to_datetime(test_nan_df.date)
    return [((train_df, test_df, train_nan_df, test_nan_df), None)]


def get_dataset(nan_percent: str):
    main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/smart_star_hourly_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly.csv"))

    main_df.date = pd.to_datetime(main_df.date)
    main_df_with_nan.date = pd.to_datetime(main_df_with_nan.date)
    return main_df, main_df_with_nan


def get_complete_dataset(nan_percent: str):
    modified_main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/"
                                                        f"smart_star_hourly_fully_modified_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly.csv"))

    main_df.date = pd.to_datetime(main_df.date)
    modified_main_df_with_nan["date"] = main_df.date
    return main_df, modified_main_df_with_nan


def get_dataset_date_modified(nan_percent: str):
    main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/smart_star_hourly_date_modified_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly_date_modified.csv"))

    return main_df, main_df_with_nan


def get_dataset_irish():
    main_df = pd.read_csv(Path("/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"))
    main_df.date = pd.to_datetime(main_df.date)
    return main_df


def get_dataset_fully_modified_date_auto(nan_percent: str):
    main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/smart_star_hourly_fully_modified_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly_fully_modified.csv"))
    main_df_with_nan["real_usage"] = main_df["usage"]
    return main_df_with_nan


def get_dataset_auto(nan_percent: str):
    main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/smart_star_hourly_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly.csv"))

    main_df.date = pd.to_datetime(main_df.date)
    main_df_with_nan.date = pd.to_datetime(main_df_with_nan.date)
    main_df_with_nan["real_usage"] = main_df["usage"]
    return main_df_with_nan


def get_dataset_fully_modified_date(nan_percent: str):
    main_df_with_nan = pd.read_csv(Path(root + f"datasets/with_nan/smart_star_hourly_fully_modified_{nan_percent}.csv"))
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly_fully_modified.csv"))

    return main_df, main_df_with_nan


def generate_small_pandas_dataset():
    main_df = pd.read_csv(Path(root + "datasets/smart_star_hourly.csv"))
    main_df.date = pd.to_datetime(main_df.date)
    users_id = main_df.id.unique()
    random_ids = np.random.choice(users_id, int(len(users_id) * 0.3))
    main_df = main_df[main_df.id.isin(random_ids)]
    main_df = main_df.reset_index(drop=True)
    return main_df




