import pandas as pd

if __name__ == '__main__':
    main_df = pd.read_csv("./../../datasets/smart_star.csv")
    main_df.date = pd.to_datetime(main_df.date)
    main_df = main_df.set_index("date")
    main_df = main_df.groupby("id").apply(lambda x: x.resample("1H").sum()[["usage"]]).reset_index()
    main_df.to_csv("./../../datasets/smart_star_hourly.csv", index=False)
