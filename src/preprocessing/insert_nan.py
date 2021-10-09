import pandas as pd
import numpy as np

nan_percents = [0.01, 0.05]

if __name__ == '__main__':
    file_name = "smart_star_small_date_modified"
    main_df = pd.read_csv("../../datasets/{}.csv".format(file_name))

    for percent in nan_percents:
        record_count = main_df.shape[0]
        random_index = np.random.choice(range(record_count), int(record_count * percent), replace=False)
        main_df.loc[random_index, "usage"] = np.nan
        print("nan percent = ", percent)
        print(main_df.isna().sum())
        main_df.to_csv("../../datasets/with_nan/{}_{}.csv".format(file_name, percent), index=False)
