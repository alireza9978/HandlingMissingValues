import datetime
import os
import numpy as np
import pandas as pd

from src.preprocessing.load_dataset import root


def convert_users_users():
    data_columns = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
                    'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
                    'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'SysABP', 'Temp', 'TroponinT', 'Urine', 'WBC',
                    'Weight', 'pH']
    main_df = pd.DataFrame()
    root_path = root + "datasets/PhysioNet/set-a/"
    for item in os.listdir(root_path):
        if item.endswith(".txt"):
            user_id = item.split(".")[0]
            temp_df = pd.DataFrame(index=np.arange(0, 48 * 60), columns=data_columns)
            item_path = root_path + item
            f = open(item_path, "r")
            file_lines = f.readlines()
            for line in file_lines[7:]:
                time, feature_name, value = line.strip().split(",")
                hours, minutes = time.split(":")
                index = int(hours) * 60 + int(minutes)
                temp_df.loc[index, feature_name] = value
            temp_df.index = temp_df.index.to_series().apply(lambda x: pd.to_timedelta(x, unit="minute"))
            temp_df["user_id"] = user_id
            main_df = main_df.append(temp_df)
    main_df.to_csv(root + "datasets/PhysioNet/dataset.csv")


if __name__ == '__main__':
    convert_users_users()
