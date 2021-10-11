import numpy as np
import pandas as pd

from src.preprocessing.convert_timestamps import convert_date

root = "/home/alireza/projects/python/HandlingMissingValues/"


def get_dataset():
    main_df_with_nan = pd.read_csv(root + "datasets/with_nan/smart_star_small_0.01.csv")
    main_df = pd.read_csv(root + "datasets/smart_star_small.csv")

    main_df.date = pd.to_datetime(main_df.date)
    main_df_with_nan.date = pd.to_datetime(main_df_with_nan.date)
    return main_df, main_df_with_nan


def get_dataset_with_modified_date():
    main_df_with_nan = pd.read_csv(root + "datasets/with_nan/smart_star_small_date_modified_0.01.csv")
    main_df = pd.read_csv(root + "datasets/smart_star_small_date_modified.csv")

    return main_df, main_df_with_nan


def get_dataset_fully_modified_date():
    main_df_with_nan = pd.read_csv(root + "datasets/with_nan/smart_star_small_fully_modified_0.01.csv")
    main_df = pd.read_csv(root + "datasets/smart_star_small_fully_modified.csv")

    return main_df, main_df_with_nan


def generate_small_pandas_dataset():
    main_df = pd.read_csv(root + "datasets/smart_star_hourly.csv")
    main_df.date = pd.to_datetime(main_df.date)
    users_id = main_df.id.unique()
    random_ids = np.random.choice(users_id, int(len(users_id) * 0.3))
    main_df = main_df[main_df.id.isin(random_ids)]
    main_df = main_df.reset_index(drop=True)
    return main_df


def load_weather_dataset():
    weather_df = pd.read_csv(root + "datasets/smart_star_weather.csv")
    weather_df.date = pd.to_datetime(weather_df.date)
    weather_df = weather_df.drop(columns=["icon", "summary", "time"])
    weather_df = weather_df.set_index("date").resample("1H").mean().reset_index()
    return weather_df


def add_holiday_weather_convert_date():
    main_df = pd.read_csv(root + "datasets/smart_star_small.csv")
    main_df.date = pd.to_datetime(main_df.date)
    holiday_df = pd.read_csv(root + "datasets/holiday.csv")
    holiday_df["date"] = pd.to_datetime(holiday_df.date)
    holiday_df = holiday_df.set_index("date")
    main_df["only_date"] = pd.to_datetime(main_df.date.dt.date)
    main_df = main_df.set_index("only_date").join(holiday_df).reset_index(drop=True)

    weather_df = load_weather_dataset()
    weather_df = weather_df.set_index("date")
    main_df = main_df.set_index("date").join(weather_df).reset_index(drop=False)

    main_df = convert_date(main_df)

    main_df.to_csv(root + "datasets/smart_star_small_fully_modified.csv", index=False)


if __name__ == '__main__':
    add_holiday_weather_convert_date()
