import numpy as np
import pandas as pd

root = "C:/Users/Alireza/PycharmProjects/HandlingMissingValues/"


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


def generate_small_pandas_dataset():
    main_df = pd.read_csv("C:/Users/Alireza/PycharmProjects/HandlingMissingValues/datasets/smart_star_hourly.csv")
    main_df.date = pd.to_datetime(main_df.date)
    users_id = main_df.id.unique()
    random_ids = np.random.choice(users_id, int(len(users_id) * 0.3))
    main_df = main_df[main_df.id.isin(random_ids)]
    main_df = main_df.reset_index(drop=True)
    return main_df
