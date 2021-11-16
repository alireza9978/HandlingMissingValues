import multiprocessing

import pandas as pd
import swifter
from joblib import Parallel, delayed

a = swifter.config


def apply_parallel(data_frame_grouped, func, args=None):
    if args is None:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()/2))(
            delayed(func)(group) for name, group in data_frame_grouped)
        return pd.DataFrame(result_list)
    else:
        result_list = Parallel(n_jobs=int(multiprocessing.cpu_count()/2))(
            delayed(func)(group, args) for name, group in data_frame_grouped)
        return pd.DataFrame(result_list)
