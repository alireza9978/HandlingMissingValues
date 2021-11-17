import numpy as np
import pandas as pd


def get_random_user(x: pd.DataFrame, x_nan: pd.DataFrame):
    user_ids = x.id.unique()
    temp_id = np.random.choice(user_ids, 1)[0]
    x = x[x.id == temp_id]
    x = x.reset_index(drop=True)
    x_nan = x_nan[x_nan.id == temp_id]
    x_nan = x_nan.reset_index(drop=True)
    return x, x_nan
