import multiprocessing

import pandas as pd
import swifter
from joblib import Parallel, delayed

a = swifter.config


def apply_parallel(data_frame_grouped, func):
    result_list = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(func)(group) for name, group in data_frame_grouped)
    return pd.DataFrame(result_list)
