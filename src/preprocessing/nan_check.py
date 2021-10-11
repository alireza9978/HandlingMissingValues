import os

import pandas as pd

root = "/home/alireza/projects/python/HandlingMissingValues/"


def method_one():
    main_df = pd.read_csv(root + "datasets/smart_star.csv")
    main_df.date = pd.to_datetime(main_df.date)
    nan_df = main_df.groupby("id").apply(lambda x: x.usage.isna().sum())
    nan_df.to_csv(root + "results/nan.csv")


def method_two():
    root_path = root + "datasets/apartment/"
    years = ["2014", "2015", "2016"]
    for year in years:
        for item in os.listdir(root_path + year):
            if item.endswith(".csv"):
                item_path = root_path + year + "/" + item
                temp_df = pd.read_csv(item_path, header=None)
                temp_df.columns = ["date", "usage"]
                if temp_df.usage.isna().sum() != 0:
                    print(item_path)


def method_two_weather():
    root_path = root + "datasets/apartment-weather/"
    years = ["2014", "2015", "2016"]
    for year in years:
        file_name = "apartment{}.csv".format(year)
        item_path = root_path + file_name
        temp_df = pd.read_csv(item_path)
        if temp_df.isna().sum().sum() != 0:
            print(file_name)
            print(temp_df.isna().sum())


if __name__ == '__main__':
    method_two_weather()
