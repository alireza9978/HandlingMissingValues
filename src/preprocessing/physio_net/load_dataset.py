import pandas as pd
import ujson as json

from src.preprocessing.load_dataset import root

data_columns = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2',
                'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
                'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
                'Urine', 'WBC', 'Weight', 'pH']


def save_sample_user():
    df = pd.read_csv(root + "datasets/physionet/dataset.csv")
    print(df)
    a = df[df.user_id == df.user_id[0]]
    a.to_excel("physionet_sample_user.xlsx")


def load_brits_json():
    json_content = open(root + 'other_methods/BRITS/json/json').readlines()
    rec = json.loads(json_content[5])
    print(rec)


if __name__ == '__main__':
    load_brits_json()
