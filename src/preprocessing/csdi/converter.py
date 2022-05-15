import os

import numpy as np
import pandas as pd
from src.preprocessing.smart_star.load_dataset import root



def make_same_length(temp_df: pd.DataFrame):
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df["only_date"] = temp_df.date.dt.date
    data_count = temp_df[["only_date", "id", "usage"]].groupby(["id", "only_date"]).count()
    bad_user_day = data_count[data_count.usage != 24].reset_index()
    for _, row in bad_user_day.iterrows():
        temp_df = temp_df[~((temp_df.id == row.id) & (temp_df.only_date == row.only_date))]

    gr = temp_df[["date", "id"]].groupby("id")
    start = gr.min().max().date
    end = gr.max().min().date

    temp_df = temp_df[(temp_df.date >= pd.to_datetime(start)) & (pd.to_datetime(end) >= temp_df.date)]
    # temp_df.to_csv(Path(root + ""), index=False)
    return temp_df, temp_df[temp_df.id == 1].shape[0]


def create_id_list(temp_df: pd.DataFrame):
    ids = temp_df.id.unique()
    np.save(root + "other_methods/CSDI/data/smart_star/id_lists", ids)


def create_dataset(temp_df: pd.DataFrame):
    user_id = np.load(root + "other_methods/CSDI/data/smart_star/id_lists.npy")
    os.makedirs(root + f"other_methods/CSDI/data/smart_star/users", exist_ok=True)
    temp_df, _ = make_same_length(temp_df)
    for id_ in user_id:
        user_df = temp_df[temp_df.id == id_].drop(columns=["id", "date", "only_date"])
        user_df.to_csv(root + f"other_methods/CSDI/data/smart_star/users/{id_}.csv", index=False)


def save_single_user(temp_df: pd.DataFrame):
    row_0 = temp_df.iloc[0]
    temp_df = temp_df.drop(columns=["id", "date", "week"])
    id_ = f"{row_0.id}".rjust(3, "0") + f"{row_0.year}".rjust(4, "0") + f"{row_0.week}".rjust(2, "0")
    temp_df.to_csv(root + f"other_methods/CSDI/data/smart_star/users/{id_}.csv", index=False)
    return id_


def create_dataset_weekly(temp_df: pd.DataFrame):
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df["week"] = temp_df.date.dt.isocalendar().week
    result = temp_df.groupby(["id", "year", "week"]).apply(save_single_user)
    np.save(root + "other_methods/CSDI/data/smart_star/id_lists", result.values.astype(np.str))


if __name__ == '__main__':
    file_path = "datasets/smart_star/smart_star_hourly_final_with_date.csv"
    main_df = pd.read_csv(root + file_path)
    create_dataset_weekly(main_df)
