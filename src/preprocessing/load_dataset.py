import pandas as pd


def get_dataset():
    main_df = pd.read_csv("../../datasets/with_nan/smart_star_small_0.01.csv")
    main_df.date = pd.to_datetime(main_df.date)
    return main_df
