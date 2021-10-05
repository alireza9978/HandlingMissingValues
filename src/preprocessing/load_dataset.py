import pandas as pd


def get_dataset():
    main_df_with_nan = pd.read_csv(
        "C:/Users/Alireza/PycharmProjects/HandlingMissingValues/datasets/with_nan/smart_star_small_0.01.csv")
    main_df = pd.read_csv("C:/Users/Alireza/PycharmProjects/HandlingMissingValues/datasets/smart_star_small.csv")

    main_df.date = pd.to_datetime(main_df.date)
    main_df_with_nan.date = pd.to_datetime(main_df_with_nan.date)
    return main_df.usage.to_numpy().reshape(-1, 1), main_df_with_nan.usage.to_numpy().reshape(-1, 1)
