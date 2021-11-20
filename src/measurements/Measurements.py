import numpy as np
import pandas as pd


def root_mean_square_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff2 = np.square(diff)
    diff2_mean = np.mean(diff2)
    root_diff2_mean = np.sqrt(diff2_mean)
    return root_diff2_mean, diff2


def mean_square_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff2 = np.square(diff)
    diff2_mean = np.mean(diff2)
    return diff2_mean, diff2


def mean_absolute_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff_abs = np.abs(diff)
    diff_abs_mean = np.mean(diff_abs)
    return diff_abs_mean, diff_abs


def mean_absolute_percentage_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    not_zero_index = np.where(x != 0)[0]
    diff = diff[not_zero_index]
    x = x[not_zero_index]
    diff_percent = np.divide(diff, x)
    diff_percent_abs = np.abs(diff_percent)
    diff_percent_abs_mean = np.mean(diff_percent_abs)
    return np.multiply(diff_percent_abs_mean, 100), diff_percent_abs


def evaluate_dataframe_two(temp_df: pd.DataFrame, evaluation_function):
    error, error_column = evaluation_function(temp_df["usage"], temp_df["predicted_usage"])
    temp_df["error"] = error_column
    return error, temp_df


def evaluate_dataframe(temp_df: pd.DataFrame, evaluation_function):
    import swifter
    _ = swifter.config

    def inner_process(user_df):
        filled_value = user_df[0]
        filled_index = user_df[1]
        real_values = user_df[2].usage[filled_index].to_numpy().reshape(-1, 1)
        return evaluation_function(real_values, filled_value)

    # users_error = temp_df.swifter.progress_bar(False).apply(inner_process, axis=1)
    users_error = temp_df.apply(inner_process, axis=1)
    return users_error.mean()


def calculate_measures(x: np.ndarray, x_filled_nan: np.ndarray):
    print("mean_square_error", mean_square_error(x, x_filled_nan))
    print("mean_absolute_error", mean_absolute_error(x, x_filled_nan))
    print("mean_absolute_percentage_error", mean_absolute_percentage_error(x, x_filled_nan))


if __name__ == '__main__':
    filled = np.array([1, 2.5, 3, 4.5, 1]).reshape(-1, 1)
    real = np.array([1, 2, 3, 4, 0]).reshape(-1, 1)
    # filled_x = np.array([1, 2, 2.5, 4])
    # test_x = np.array([1, 2])
    # test_x_nan = np.array([np.nan])
    # filled_x = np.array([1.5, 2])
    calculate_measures(real, filled)
