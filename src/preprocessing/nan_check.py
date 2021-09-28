import pandas as pd
import os


def method_one():
    main_df = pd.read_csv("../../datasets/smart_star.csv")
    main_df.date = pd.to_datetime(main_df.date)
    nan_df = main_df.groupby("id").apply(lambda x: x.usage.isna().sum())
    nan_df.to_csv("../../results/nan.csv")


def method_two():
    root_path = "./../../datasets/apartment/"
    years = ["2014", "2015", "2016"]
    for year in years:
        for item in os.listdir(root_path + year):
            if item.endswith(".csv"):
                item_path = root_path + year + "/" + item
                temp_df = pd.read_csv(item_path, header=None)
                temp_df.columns = ["date", "usage"]
                if temp_df.usage.isna().sum() != 0:
                    print(item_path)


if __name__ == '__main__':
    method_two()
