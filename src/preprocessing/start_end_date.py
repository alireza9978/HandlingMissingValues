import pandas as pd

from src.preprocessing.load_dataset import load_weather_dataset

root = "/home/alireza/projects/python/HandlingMissingValues/"


def save_start_date(temp_df: pd.DataFrame, name: str):
    start_date = temp_df.groupby("id").apply(lambda x: x.date.min())
    start_date.to_csv(root + "results/{}.csv".format(name))
    return start_date


def save_end_date(temp_df: pd.DataFrame, name: str):
    end_date = temp_df.groupby("id").apply(lambda x: x.date.max())
    end_date.to_csv(root + "results/{}.csv".format(name))
    return end_date


def make_smaller_dataset():
    main_df = pd.read_csv(root + "datasets/smart_star_hourly.csv")
    main_df.date = pd.to_datetime(main_df.date)
    start_date_df = save_start_date(main_df)
    end_date_df = save_end_date(main_df)
    a = main_df.date > pd.to_datetime(start_date_df.max())
    b = main_df.date < pd.to_datetime(end_date_df.min())
    small_df = main_df[a & b]
    small_df.to_csv(root + "datasets/smart_star_small.csv", index=False)


if __name__ == '__main__':
    weather_df = load_weather_dataset()
    weather_df["id"] = 1
    save_end_date(weather_df, "weather_end")
    save_start_date(weather_df, "weather_start")
