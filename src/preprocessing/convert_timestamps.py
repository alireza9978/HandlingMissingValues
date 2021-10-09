import pandas as pd
import numpy as np
from load_dataset import generate_small_pandas_dataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

if __name__ == '__main__':
    temp_df = generate_small_pandas_dataset()

    temp_df["day"] = temp_df.date.dt.day
    temp_df["month"] = temp_df.date.dt.month
    temp_df["year"] = temp_df.date.dt.year
    temp_df["hour"] = temp_df.date.dt.hour
    temp_df["day_of_week"] = temp_df.date.dt.dayofweek
    temp_df["season"] = temp_df["month"].apply(lambda x: int((((x + 1) % 12) - (((x + 1) % 12) % 3)) / 3))
    encoder = OneHotEncoder()
    season_one_hot_encode = encoder.fit_transform(temp_df["season"].to_numpy().reshape(-1, 1)).toarray()
    seasons_df = pd.DataFrame(season_one_hot_encode, columns=["winter", "spring", "summer", "fall"])
    temp_df = temp_df.join(seasons_df)
    temp_df = temp_df.drop(columns=["season"])
    selected_columns = ["day", "month", "hour", "day_of_week"]
    for column in selected_columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        temp_df[column] = scaler.fit_transform(temp_df[column].to_numpy().reshape(-1, 1)) * np.pi
        temp_df[column + "_x"] = np.sin(temp_df[column])
        temp_df[column + "_y"] = np.cos(temp_df[column])

    temp_df = temp_df.drop(columns=selected_columns + ["date"])
    temp_df.to_csv(
        "C:/Users/Alireza/PycharmProjects/HandlingMissingValues/datasets/smart_star_small_date_modified.csv",
        index=False)
