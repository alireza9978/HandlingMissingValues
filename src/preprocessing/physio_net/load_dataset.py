import pandas as pd

from src.preprocessing.load_dataset import root

df = pd.read_csv(root + "datasets/physionet/dataset.csv")
print(df)
a = df[df.user_id == df.user_id[0]]
a.to_excel("physionet_sample_user.xlsx")
