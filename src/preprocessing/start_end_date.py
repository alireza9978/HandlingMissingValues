import pandas as pd


def save_start_date(temp_df: pd.DataFrame):
    start_date = temp_df.groupby("id").apply(lambda x: x.date.min())
    start_date.to_csv("../../results/start_date.csv")
    return start_date


def save_end_date(temp_df: pd.DataFrame):
    end_date = temp_df.groupby("id").apply(lambda x: x.date.max())
    end_date.to_csv("../../results/end_date.csv")
    return end_date


if __name__ == '__main__':
    main_df = pd.read_csv("../../datasets/smart_star_hourly.csv")
    main_df.date = pd.to_datetime(main_df.date)
    start_date_df = save_start_date(main_df)
    end_date_df = save_end_date(main_df)
    a = main_df.date > pd.to_datetime(start_date_df.max())
    b = main_df.date < pd.to_datetime(end_date_df.min())
    small_df = main_df[a & b]
    small_df.to_csv("../../datasets/smart_star_small.csv", index=False)
