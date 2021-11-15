from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.load_dataset import root

nan_percents = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
nan_percents_str = ["0.01", "0.05", "0.1", "0.15", "0.2",
                    "0.25", "0.3", "0.35", "0.4", "0.45", "0.5"]

if __name__ == '__main__':
    file_name = "smart_star_hourly_date_modified"
    main_df = pd.read_csv(Path(root + "datasets/{}.csv".format(file_name)))

    for percent in nan_percents:
        record_count = main_df.shape[0]
        random_index = np.random.choice(range(record_count), int(record_count * percent), replace=False)
        main_df.loc[random_index, "usage"] = np.nan
        print("nan percent = ", percent)
        print(main_df.isna().sum())
        main_df.to_csv(Path(root + "datasets/with_nan/{}_{}.csv".format(file_name, percent)), index=False)
