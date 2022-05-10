import pandas as pd

from src.preprocessing.smart_star.load_dataset import root


def load_air_dfs():
    train_df = pd.read_csv(root + "datasets/air/train_x.csv", parse_dates=['date'], date_parser=pd.to_datetime)
    test_df = pd.read_csv(root + "datasets/air/test_x.csv", parse_dates=['date'], date_parser=pd.to_datetime)
    train_nan_df = pd.read_csv(root + "datasets/air/train_x_nan.csv", parse_dates=['date'],
                               date_parser=pd.to_datetime)
    test_nan_df = pd.read_csv(root + "datasets/air/test_x_nan.csv", parse_dates=['date'],
                              date_parser=pd.to_datetime)

    data_frames = ((train_df, test_df, train_nan_df, test_nan_df), None)
    return data_frames
