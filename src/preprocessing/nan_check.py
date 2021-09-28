import pandas as pd

if __name__ == '__main__':
    main_df = pd.read_csv("../../datasets/smart_star.csv")
    main_df.date = pd.to_datetime(main_df.date)
    nan_df = main_df.groupby("id").apply(lambda x: x.usage.isna().sum())
    nan_df.to_csv("../../results/nan.csv")
