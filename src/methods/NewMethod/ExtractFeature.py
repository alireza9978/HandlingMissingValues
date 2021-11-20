import math
import pandas as pd

from src.preprocessing.load_dataset import get_dataset
from src.utils.Dataset import get_random_user, load_error
from src.utils.Methods import method_name_single_feature
from src.utils.Methods import measures_name


def calculate_feature(temp_df: pd.DataFrame, window_size: int):
    import swifter
    _ = swifter.config

    minimum_index = temp_df.index[0]
    maximum_index = temp_df.index[-1]

    nan_indexes = temp_df.usage.isna()
    result_df = pd.DataFrame()
    for temp_index in nan_indexes[nan_indexes].index:
        temp_minimum_index = max((temp_index - window_size, minimum_index))
        temp_maximum_index = min((temp_index + window_size, maximum_index))
        left_feature = temp_df.loc[temp_minimum_index:temp_index - 1].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        left_feature.index += "_left"
        right_feature = temp_df.loc[temp_index + 1:temp_maximum_index].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        right_feature.index += "_right"
        feature_row = pd.concat([left_feature, right_feature])
        feature_row.name = temp_index
        result_df = result_df.append(feature_row)

    return result_df


if __name__ == '__main__':
    nan_percent = "0.05"
    x, x_nan = get_dataset(nan_percent)
    (x, x_nan) = get_random_user(x, x_nan)
    moving_features = x_nan.groupby("id").apply(calculate_feature, 12)
    moving_features = moving_features.reset_index(level=0)
    for name in method_name_single_feature:
        error_df = load_error(nan_percent, name, measures_name[0])
        temp_columns = error_df.columns.to_list()
        temp_columns[-1] = name
        error_df.columns = temp_columns
        moving_features = moving_features.join(error_df[[name, "index"]].set_index("index"))
    print(moving_features)
    print(moving_features.columns)
    print(moving_features.isna().sum())
