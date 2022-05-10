import numpy as np
import pandas as pd
from pandas import DateOffset

from src.preprocessing.smart_star.load_dataset import root

file_path = root + "datasets/power/uci/household_power_consumption.txt"
target_file_path = root + "datasets/power/uci/dataset.csv"


def clean_dataset():
    temp_df = pd.read_csv(file_path, delimiter=";", low_memory=False)
    temp_df["date"] = pd.to_datetime(temp_df["Date"]) + pd.to_timedelta(temp_df["Time"])
    temp_df.drop(columns=["Date", "Time"], inplace=True)
    for temp_col in temp_df.columns:
        temp_df.loc[temp_df[temp_col] == '?', temp_col] = np.nan
    target_df = pd.DataFrame(
        index=pd.date_range(temp_df["date"].min(), temp_df["date"].max(), freq=DateOffset(minuets=1)))
    target_df = target_df.join(temp_df.set_index("date"))
    target_df.to_csv(target_file_path, index_label="date")


def load_dataset():
    temp_df = pd.read_csv(target_file_path, index_col="date", date_parser=pd.to_datetime, parse_dates=["date"])
    temp_df["id"] = 1
    return temp_df


if __name__ == '__main__':
    load_dataset()
