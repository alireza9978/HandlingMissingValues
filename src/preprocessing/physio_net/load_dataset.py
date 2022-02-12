import pandas as pd

from src.preprocessing.load_dataset import root

df = pd.read_csv(root + "datasets/physionet/dataset.csv")
print(df)
