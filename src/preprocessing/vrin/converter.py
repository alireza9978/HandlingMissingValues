import pickle

import numpy as np
import pandas as pd
import swifter

from src.preprocessing.smart_star.load_dataset import root

_ = swifter.config
hours_index = [f"{i}" for i in range(24)]


def make_same_length(temp_df: pd.DataFrame):
    temp_df.date = pd.to_datetime(temp_df.date)
    temp_df["only_date"] = temp_df.date.dt.date
    data_count = temp_df[["only_date", "id", "usage"]].groupby(["id", "only_date"]).count()
    bad_user_day = data_count[data_count.usage != 24].reset_index()
    for _, row in bad_user_day.iterrows():
        temp_df = temp_df[~((temp_df.id == row.id) & (temp_df.only_date == row.only_date))]

    gr = temp_df[["date", "id"]].groupby("id")
    start = gr.min().max().date
    end = gr.max().min().date

    temp_df = temp_df[(temp_df.date >= pd.to_datetime(start)) & (pd.to_datetime(end) >= temp_df.date)]
    # temp_df.to_csv(Path(root + ""), index=False)
    return temp_df, temp_df[temp_df.id == 1].shape[0]


def create_delta_column(temp_col: pd.Series):
    count = 1
    deltas = [0]
    for i in range(1, temp_col.shape[0]):
        if temp_col[i]:
            deltas.append(count)
            count = 1
        else:
            deltas.append(count)
            count += 1
    return pd.Series(deltas)


def make_rows(temp_df: pd.DataFrame):
    if temp_df.shape[0] == 24:
        return temp_df.usage.to_numpy()
        # return pd.Series(temp_df.usage.values, index=hours_index, name=temp_df.iloc[0].only_date)
    else:
        return np.nan


def smart_star():
    file_path = "datasets/smart_star/smart_star_hourly_final_with_date.csv"
    temp_df = pd.read_csv(root + file_path)
    data_columns = ['usage', 'holiday', 'weekend', 'temperature', 'humidity',
                    'visibility', 'apparentTemperature', 'pressure', 'windSpeed',
                    'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint',
                    'precipProbability', 'year', 'winter', 'spring', 'summer', 'fall',
                    'day_x', 'day_y', 'month_x', 'month_y', 'hour_x', 'hour_y',
                    'day_of_week_x', 'day_of_week_y']
    temp_df, length = make_same_length(temp_df)
    users_id = temp_df.id.unique()

    temp_user_ids_array = np.random.choice(users_id, (3, int(np.floor(users_id.shape[0] / 3))), replace=False)
    final_array = []
    for temp_user_ids in temp_user_ids_array:
        temp_array = []
        for temp_user_id in temp_user_ids:
            user_df = temp_df[temp_df.id == temp_user_id]
            user_df = user_df.reset_index(drop=True)
            values = user_df[data_columns].to_numpy()
            temp_array.append(values)
        final_array.append(temp_array)

    final_array = np.array(final_array)
    final_array = final_array.reshape(
        (1, final_array.shape[0], final_array.shape[1], final_array.shape[2], final_array.shape[3]))
    with open("../../../other_methods/V-RIN/data_nan.p", 'wb') as handle:
        pickle.dump(final_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("../../../other_methods/V-RIN/label.p", 'wb') as handle:
        pickle.dump(np.zeros((1, final_array.shape[1], final_array.shape[2])), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    smart_star()
