import pandas as pd
import os

root_path = "./../../datasets/apartment/"
years = ["2014", "2015", "2016"]
main_df = pd.DataFrame()
for year in years:
    for item in os.listdir(root_path + year):
        if item.endswith(".csv"):
            item_path = root_path + year + "/" + item
            temp_df = pd.read_csv(item_path, header=None)
            temp_df.columns = ["date", "usage"]
            temp_df.date = pd.to_datetime(temp_df.date)
            temp_df["id"] = item[3:-4]
            main_df = main_df.append(temp_df)

main_df = main_df.sort_values(by=["id", "date"])
main_df.to_csv("./../../datasets/smart_star.csv", index=False)
