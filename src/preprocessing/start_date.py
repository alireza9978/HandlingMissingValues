import pandas as pd

if __name__ == '__main__':
    main_df = pd.read_csv("../../datasets/smart_star_hourly.csv")
    main_df.date = pd.to_datetime(main_df.date)
    start_date = main_df.groupby("id").apply(lambda x: x.date.min())
    start_date.to_csv("../../results/start_date.csv")
