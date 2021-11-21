from pathlib import Path

import pandas as pd

from src.preprocessing.load_dataset import root


def generate_train_test():
    file_name = "smart_star_hourly_fully_modified"
    percent = 0.3
    temp_df = pd.read_csv(Path(root + "datasets/{}.csv".format(file_name)))

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
    train_df.to_csv(Path(root + "datasets/train_test/{}_train_{}.csv".format(file_name, percent)), index=False)
    test_df.to_csv(Path(root + "datasets/train_test/{}_test_{}.csv".format(file_name, percent)), index=False)


if __name__ == '__main__':
    generate_train_test()
