import datetime
import os
import numpy as np
import pandas as pd
import swifter

from src.preprocessing.load_dataset import root

_ = swifter.config


def convert_users_users():
    main_df = pd.DataFrame()
    root_path = root + "datasets/physionet/downloaded/set-a/"
    items = os.listdir(root_path)
    count = len(items)
    for index, item in enumerate(items):
        if item.endswith(".txt"):
            user_id = item.split(".")[0]
            item_path = root_path + item
            f = open(item_path, "r")
            file_lines = f.readlines()
            temp_df = pd.DataFrame(file_lines[7:])
            if temp_df.shape[0] > 0:
                temp_cleaned_values = temp_df.swifter.progress_bar(False).apply(lambda x: str(x[0]).strip().split(","),
                                                                                axis=1)
                temp_df = pd.DataFrame(temp_cleaned_values.to_list(), columns=["time", "name", "value"])
                temp_df["value"] = temp_df["value"].astype(float)
                temp_df["name"] = temp_df["name"].astype("category")
                temp_df["user_id"] = int(user_id)
                temp_df["hour"] = temp_df["time"].swifter.progress_bar(False).apply(lambda x: int(x[:2]))
                temp_df["minute"] = temp_df["time"].swifter.progress_bar(False).apply(lambda x: int(x[3:]))
                main_df = main_df.append(temp_df[["hour", "minute", "user_id", "name", "value"]])
            print("progress: ", index, count)
    main_df.to_csv(root + "datasets/physionet/dataset.csv", index=False)


if __name__ == '__main__':
    convert_users_users()
