import numpy as np


def mean_square_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff2 = np.square(diff)
    diff2_mean = np.mean(diff2)
    root_diff2_mean = np.sqrt(diff2_mean)
    return root_diff2_mean


def mean_absolute_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    diff_abs = np.abs(diff)
    diff_abs_mean = np.mean(diff_abs)
    return diff_abs_mean


def mean_absolute_percentage_error(x: np.ndarray, x_filled_nan: np.ndarray):
    diff = x - x_filled_nan
    not_zero_index = np.where(x != 0)[0]
    diff = diff[not_zero_index]
    x = x[not_zero_index]
    diff_percent = np.divide(diff, x)
    diff_percent_abs = np.abs(diff_percent)
    diff_percent_abs_mean = np.mean(diff_percent_abs)
    return np.multiply(diff_percent_abs_mean, 100)


def calculate_measures(x: np.ndarray, x_filled_nan: np.ndarray):
    print("mean_square_error", mean_square_error(x, x_filled_nan))
    print("mean_absolute_error", mean_absolute_error(x, x_filled_nan))
    print("mean_absolute_percentage_error", mean_absolute_percentage_error(x, x_filled_nan))


if __name__ == '__main__':
    # test_x = np.array([1, 2, 3, 4])
    # test_x_nan = np.array([1, 2, np.nan, 4])
    # filled_x = np.array([1, 2, 2.5, 4])
    test_x = np.array([1, 2])
    # test_x_nan = np.array([np.nan])
    filled_x = np.array([1.5, 2])
    calculate_measures(test_x, filled_x)
