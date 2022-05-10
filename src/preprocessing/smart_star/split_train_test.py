from pathlib import Path

import pandas as pd

from src.preprocessing.smart_star.load_dataset import root


def generate_train_test():
    file_name = "smart_star_hourly_final"
    percent = 0.3
    temp_df = pd.read_csv(Path(root + "datasets/smart_star/{}.csv".format(file_name)))

    def inner_split(inner_df: pd.DataFrame, train_percent: float):
        total_count = inner_df.shape[0]
        starting_index = inner_df.index[0]
        ending_index = inner_df.index[-1]
        test_start_index = ending_index - int(total_count * train_percent)
        inner_df.loc[starting_index:test_start_index - 1, "train"] = True
        inner_df.loc[test_start_index:ending_index, "train"] = False
        return inner_df

    temp_df = temp_df.groupby("id").apply(inner_split, percent)
    train_df = temp_df[temp_df.train].copy().drop(columns=["train"]).reset_index(drop=True)
    test_df = temp_df[temp_df.train == False].copy().drop(columns=["train"]).reset_index(drop=True)
    train_df.to_csv(Path(root + "datasets/smart_star/train_test/{}_train_{}.csv".format(file_name, percent)), index=False)
    test_df.to_csv(Path(root + "datasets/smart_star/train_test/{}_test_{}.csv".format(file_name, percent)), index=False)


def generate_train_test_triple():
    file_name = "smart_star_hourly"
    percent = 1 / 3
    percent_name = "0.33"
    temp_df = pd.read_csv(Path(root + "datasets/{}.csv".format(file_name)))

    def inner_split_a(inner_df: pd.DataFrame, train_percent: float):
        total_count = inner_df.shape[0]
        starting_index = inner_df.index[0]
        ending_index = inner_df.index[-1]
        test_start_index = ending_index - int(total_count * train_percent)
        inner_df.loc[starting_index:test_start_index - 1, "train"] = True
        inner_df.loc[test_start_index:ending_index, "train"] = False
        return inner_df

    def inner_split_b(inner_df: pd.DataFrame, train_percent: float):
        total_count = inner_df.shape[0]
        starting_index = inner_df.index[0]
        ending_index = inner_df.index[-1]
        test_end_index = starting_index + int(total_count * train_percent)
        inner_df.loc[starting_index:test_end_index - 1, "train"] = False
        inner_df.loc[test_end_index:ending_index, "train"] = True
        return inner_df

    def inner_split_c(inner_df: pd.DataFrame, train_percent: float):
        total_count = inner_df.shape[0]
        starting_index = inner_df.index[0]
        ending_index = inner_df.index[-1]
        test_end_index = ending_index - int(total_count * train_percent)
        test_start_index = starting_index + int(total_count * train_percent)
        inner_df.loc[starting_index:test_start_index - 1, "train"] = True
        inner_df.loc[test_start_index:test_end_index - 1, "train"] = False
        inner_df.loc[test_end_index:ending_index, "train"] = True
        return inner_df

    temp_df = temp_df.groupby("id").apply(inner_split_a, percent)
    train_df = temp_df[temp_df.train].copy().drop(columns=["train"]).reset_index(drop=True)
    test_df = temp_df[temp_df.train == False].copy().drop(columns=["train"]).reset_index(drop=True)
    train_df.to_csv(Path(root + "datasets/train_test/{}_train_{}_a.csv".format(file_name, percent_name)), index=False)
    test_df.to_csv(Path(root + "datasets/train_test/{}_test_{}_a.csv".format(file_name, percent_name)), index=False)

    temp_df = temp_df.groupby("id").apply(inner_split_b, percent)
    train_df = temp_df[temp_df.train].copy().drop(columns=["train"]).reset_index(drop=True)
    test_df = temp_df[temp_df.train == False].copy().drop(columns=["train"]).reset_index(drop=True)
    train_df.to_csv(Path(root + "datasets/train_test/{}_train_{}_b.csv".format(file_name, percent_name)), index=False)
    test_df.to_csv(Path(root + "datasets/train_test/{}_test_{}_b.csv".format(file_name, percent_name)), index=False)

    temp_df = temp_df.groupby("id").apply(inner_split_c, percent)
    train_df = temp_df[temp_df.train].copy().drop(columns=["train"]).reset_index(drop=True)
    test_df = temp_df[temp_df.train == False].copy().drop(columns=["train"]).reset_index(drop=True)
    train_df.to_csv(Path(root + "datasets/train_test/{}_train_{}_c.csv".format(file_name, percent_name)), index=False)
    test_df.to_csv(Path(root + "datasets/train_test/{}_test_{}_c.csv".format(file_name, percent_name)), index=False)


if __name__ == '__main__':
    generate_train_test()
