from pathlib import Path

import pandas as pd

from src.preprocessing.smart_star.convert_timestamps import convert_date
from src.preprocessing.smart_star.load_dataset import root


def load_weather_dataset():
    weather_df = pd.read_csv(Path(root + "datasets/smart_star/smart_star_weather.csv"))
    weather_df.date = pd.to_datetime(weather_df.date)
    weather_df = weather_df.drop(columns=["icon", "summary", "time"])
    weather_df = weather_df.set_index("date").resample("1H").mean().reset_index()
    return weather_df


def load_holidays_dataset():
    holiday_df = pd.read_csv(Path(root + "datasets/holiday.csv"))
    holiday_df["date"] = pd.to_datetime(holiday_df.date)
    holiday_df = holiday_df.set_index("date")
    return holiday_df


def add_holiday_weather_convert_date():
    main_df = pd.read_csv(Path(root + "datasets/smart_star/smart_star_hourly.csv"))
    main_df.date = pd.to_datetime(main_df.date)

    holiday_df = load_holidays_dataset()
    weather_df = load_weather_dataset()

    main_df["only_date"] = pd.to_datetime(main_df.date.dt.date)
    main_df = main_df.set_index("only_date").join(holiday_df).reset_index(drop=True)

    weather_df = weather_df.ffill()
    weather_df = weather_df.set_index("date")
    main_df = main_df.set_index("date").join(weather_df).reset_index(drop=False)

    main_df = main_df.sort_values(["id", "date"])
    main_df = main_df.reset_index(drop=True)
    temp_date = main_df.date
    main_df = convert_date(main_df)

    main_df.to_csv(Path(root + "datasets/smart_star/smart_star_hourly_final.csv"), index=False)
    main_df["date"] = temp_date
    main_df.to_csv(Path(root + "datasets/smart_star/smart_star_hourly_final_with_date.csv"), index=False)


if __name__ == '__main__':
    add_holiday_weather_convert_date()
