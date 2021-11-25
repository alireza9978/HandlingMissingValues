import multiprocessing

import pandas as pd
import swifter
from joblib import Parallel, delayed

a = swifter.config


def apply_parallel_two(data_frame_grouped, func, args=None):
    if args is None:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()))(
            delayed(func)(group) for name, group in data_frame_grouped)
        return pd.Series(result_list)
    else:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()))(
            delayed(func)(group, args) for name, group in data_frame_grouped)
        return pd.Series(result_list)


def apply_parallel(data_frame_grouped, func, args=None):
    if args is None:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()))(
            delayed(func)(group) for name, group in data_frame_grouped)
        return pd.concat(result_list)
    else:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()))(
            delayed(func)(group, args) for name, group in data_frame_grouped)
        return pd.concat(result_list)
