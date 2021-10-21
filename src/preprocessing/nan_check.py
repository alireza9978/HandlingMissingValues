import os
from src.preprocessing.load_dataset import root
import pandas as pd


def method_one(path: str, name: str):
    main_df = pd.read_csv(path)
    main_df.date = pd.to_datetime(main_df.date)
    nan_df = main_df.groupby("id").apply(lambda x: x.usage.isna().sum())
    nan_df.to_csv(root + "results/{}_nan.csv".format(name))


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


def method_three(root_path, files):
    for file in files:
        item_path = root_path + file
        temp_df = pd.read_csv(item_path)
        if temp_df.isna().sum().sum() != 0:
            print(file)
            print(temp_df.isna().sum())


if __name__ == '__main__':
    irish_path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
    # smart_star_path = root + "datasets/smart_star.csv"
    # method_one(irish_path, "irish")
    # method_one(smart_star_path, "smart_star")
    # method_three("/mnt/6EFAD426FAD3E7FB/datasets/HUE/", ["Residential_{}.csv".format(i) for i in range(1, 28)])
    # method_three("/mnt/6EFAD426FAD3E7FB/datasets/Smart meter in london/files/", ["block_{}.csv".format(i) for i in range(1, 111)])
