import datetime
import os
from pathlib import Path

import pandas as pd
from src.preprocessing.smart_star.load_dataset import root

years = ["2014", "2015", "2016"]


def merge_usage():
    root_path = "/mnt/6EFAD426FAD3E7FB/datasets/smart/apartment/"
    main_df = pd.DataFrame()
    for year in years:
        for item in os.listdir(root_path + year):
            if item.endswith(".csv"):
                item_path = root_path + year + "/" + item
                print(item_path)
                temp_df = pd.read_csv(item_path, header=None)
                temp_df.columns = ["date", "usage"]
                temp_df.date = pd.to_datetime(temp_df.date)
                temp_df["id"] = item[3:-4]
                main_df = main_df.append(temp_df)

    main_df = main_df.sort_values(by=["id", "date"])
    main_df.to_csv(root + "datasets/smart_star/smart_star.csv", index=False)


def merge_weather():
    root_path = "/mnt/6EFAD426FAD3E7FB/datasets/smart/apartment-weather/"
    main_df = pd.DataFrame(pd.date_range(start='2014-01-01 08:00:00', end='2017-01-01 07:00:00', freq="1H").to_series(),
                           columns=['date'])
    main_df.date = pd.to_datetime(main_df.date)
    main_df = main_df.set_index("date")
    for year in years:
        file_name = "apartment{}.csv".format(year)
        item_path = root_path + file_name
        temp_df = pd.read_csv(Path(item_path))
        temp_df["date"] = temp_df["time"].apply(lambda x: datetime.datetime.fromtimestamp(x))
        temp_df.date = temp_df.date - pd.to_timedelta(30, unit="minute")
        temp_df["date"] = pd.to_datetime(temp_df.date)
        temp_df = temp_df.set_index("date")
        main_df.loc[temp_df.index, temp_df.columns] = temp_df
        main_df = main_df.reindex(columns=main_df.columns.union(temp_df.columns))

    main_df = main_df.ffill()
    main_df = main_df.reset_index()
    main_df.to_csv(root + "datasets/smart_star/smart_star_weather.csv", index=False)


def merge_usage_london():
    root_path = "/mnt/6EFAD426FAD3E7FB/datasets/Smart meter in london/files/"
    main_df = pd.DataFrame()
    for number in range(50, 111):
        item_path = root_path + "block_{}.csv".format(number)
        temp_df = pd.read_csv(item_path)
        temp_df.columns = ["id", "date", "usage"]
        temp_df.date = pd.to_datetime(temp_df.date)
        temp_df["id"] = temp_df["id"].apply(lambda x: int(x[3:]))
        main_df = main_df.append(temp_df)
        print(str(number) + " done")

    main_df = main_df.sort_values(by=["id", "date"])
    main_df.to_csv(root + "datasets/london_50_111.csv", index=False)


if __name__ == '__main__':
    # merge_usage()
    merge_weather()
