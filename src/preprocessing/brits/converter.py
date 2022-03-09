import numpy as np
import pandas as pd
import swifter
import ujson as json

from src.preprocessing.load_dataset import root
from src.preprocessing.physio_net.load_dataset import data_columns

_ = swifter.config


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
    for i in range(mask_df.shape[0]):
        for j in range(mask_df.shape[1]):
            if mask_df.iloc[i, j]:
                values_indexes.append((i, j))
    eval_indexes = np.random.choice(len(values_indexes), int(len(values_indexes) * 0.1), replace=False)

    evals_df = temp_df.copy()
    evals_df.loc[:, :] = np.nan
    eval_masks_df = mask_df.copy()
    eval_masks_df.loc[:, :] = False
    for eval_index in eval_indexes:
        index = values_indexes[eval_index]
        mask_df.iloc[index[0], index[1]] = False
        eval_masks_df.iloc[index[0], index[1]] = True
        evals_df.iloc[index[0], index[1]] = temp_df.iloc[index[0], index[1]]
        temp_df.iloc[index[0], index[1]] = np.nan

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
    physio()
