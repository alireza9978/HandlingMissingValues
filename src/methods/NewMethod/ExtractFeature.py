import pandas as pd
from src.preprocessing.load_dataset import get_dataset
from src.utils.Dataset import get_random_user


def calculate_feature(temp_df: pd.DataFrame):
    temp_feature = temp_df.usage.rolling(24, min_periods=1).agg(
        {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean", "usage_median": "median",
         "usage_var": "var", "usage_std": "std", "usage_skew": "skew", "usage_kurt": "kurt", "usage_count": "count"})
    return temp_feature


if __name__ == '__main__':
    x, x_nan = get_dataset("0.05")
    (x, x_nan) = get_random_user(x, x_nan)
    moving_features = x_nan.groupby("id").apply(calculate_feature)
    print(moving_features.loc[-1])