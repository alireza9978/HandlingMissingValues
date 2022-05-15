import numpy as np
import pandas as pd
import swifter
import ujson as json

from src.preprocessing.physio_net.load_dataset import data_columns
from src.preprocessing.smart_star.load_dataset import root

_ = swifter.config


def make_same_length(temp_df: pd.DataFrame):
    temp_df.date = pd.to_datetime(temp_df.date)

    gr = temp_df[["date", "id"]].groupby("id")
    start = gr.min().max().date
    end = gr.max().min().date

    temp_df = temp_df[(temp_df.date >= pd.to_datetime(start)) & (pd.to_datetime(end) >= temp_df.date)]
    # temp_df.to_csv(Path(root + ""), index=False)
    return temp_df


def read_sample_json():
    content = open(root + 'other_methods/BRITS/json/json').readlines()
    indices = np.arange(len(content))
    idx = np.random.choice(indices)
    rec = json.loads(content[idx])
    return rec


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


def convert_table_to_json(temp_df: pd.DataFrame):
    mask_df = ~temp_df.isna()

    values_indexes = []
    for i in range(mask_df.shape[1]):
        for j in range(mask_df.shape[0]):
            values_indexes.append((i, j))
    missing_value_index = []
    for i in range(mask_df.shape[1]):
        temp_col = mask_df.iloc[:, i]
        temp_col = temp_col[~temp_col]
        for j in range(temp_col.shape[0]):
            k = mask_df.index.get_loc(temp_col.index[j])
            missing_value_index.append((i, k))
    for value in reversed(missing_value_index):
        position = value[0] * mask_df.shape[0] + value[1]
        del values_indexes[position]
    eval_indexes = np.random.choice(len(values_indexes), int(len(values_indexes) * 0.1), replace=False)

    temp_df = temp_df.astype(np.float)
    evals_df = temp_df.copy()
    evals_df.loc[:, :] = float(0)
    eval_masks_df = mask_df.copy()
    eval_masks_df.loc[:, :] = False
    for eval_index in eval_indexes:
        index = values_indexes[eval_index]
        i, j = index[1], index[0]
        mask_df.iloc[i, j] = True
        eval_masks_df.iloc[i, j] = True
        evals_df.iloc[i, j] = temp_df.iloc[i, j]
        temp_df.iloc[i, j] = float(0)

    delta_df = mask_df.apply(create_delta_column)

    final_dict = {"forward": [], "backward": []}
    for i in range(temp_df.shape[0]):
        temp_dict = {"evals": evals_df.iloc[i].to_list(), "deltas": delta_df.iloc[i].to_list(),
                     "masks": mask_df.iloc[i].to_list(), "values": temp_df.iloc[i].to_list(),
                     "eval_masks": eval_masks_df.iloc[i].to_list()}
        final_dict["forward"].append(temp_dict)
    for i in range(temp_df.shape[0] - 1, -1, -1):
        temp_dict = {"evals": evals_df.iloc[i].to_list(), "deltas": delta_df.iloc[i].to_list(),
                     "masks": mask_df.iloc[i].to_list(), "values": temp_df.iloc[i].to_list(),
                     "eval_masks": eval_masks_df.iloc[i].to_list()}
        final_dict["backward"].append(temp_dict)
    return final_dict


def smart_star():
    temp_df = pd.read_csv(root + "datasets/smart_star/smart_star_hourly_final_with_date.csv")
    temp_df = make_same_length(temp_df)
    users_id = temp_df.id.unique()
    temp_user_ids = np.random.choice(users_id, 3)
    jsons = []
    for temp_user_id in temp_user_ids:
        user_df = temp_df[temp_df.id == temp_user_id]
        user_df = user_df.reset_index(drop=True)
        user_df["week"] = user_df.date.dt.isocalendar().week
        temp_count = user_df[["year", "week", "id"]].groupby(["year", "week"]).count()
        temp_count = temp_count[(temp_count == 168).id].reset_index()
        user_df = user_df.set_index("date").drop(columns=["id"])
        for index, row in temp_count.iterrows():
            temp_json = convert_table_to_json(
                user_df[(user_df.year == row.year) & (user_df.week == row.week)].drop(columns=["week"]))
            temp_json["label"] = 0
            jsons.append(json.dumps(temp_json) + "\n")

    f = open(root + "datasets/smart_star/brits.txt", "w")
    f.writelines(jsons)
    f.close()


def physio():
    temp_df = pd.read_csv(root + "datasets/physionet/dataset.csv")
    users_id = temp_df.user_id.unique()
    # temp_user_id = np.random.choice(users_id)
    # temp_df["time"] = temp_df.swifter.apply(lambda row: row["hour"] * 60 + row["minute"], axis=1)
    # temp_df = temp_df.drop(columns=['hour', 'minute', 'user_id'])
    jsons = []
    for temp_user_id in users_id:
        user_df = temp_df[temp_df.user_id == temp_user_id]
        user_df = user_df.reset_index(drop=True)
        user_df = user_df.pivot_table(index=["hour"], columns="name", values="value", aggfunc=np.mean)
        for column in list(set(data_columns).difference(user_df.columns)):
            user_df[column] = np.nan
        for i in list(set(list(range(49))).difference(user_df.index)):
            user_df.loc[i] = np.nan
        user_df = user_df.sort_index()
        temp_json = convert_table_to_json(user_df)
        jsons.append(temp_json)

    f = open(root + "datasets/physionet.txt", "w")
    f.write(str(jsons))
    f.close()


if __name__ == '__main__':
    # physio()
    smart_star()
