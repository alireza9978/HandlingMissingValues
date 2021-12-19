import pandas as pd

from src.methods.Simple.Interpolation import Interpolation
from src.preprocessing.load_dataset import root


def load_files():
    path = root + "datasets/air/pm25_ground.txt"
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            values = line.split(",")
            print(values)


def load_files_two():
    def split_train_test(inner_df):
        inner_df = inner_df.reset_index("id")
        inner_df["month"] = inner_df.index.month
        train = inner_df[~inner_df.month.isin([7, 10, 1, 4])].drop(columns=["month"])
        test = inner_df[inner_df.month.isin([7, 10, 1, 4])].drop(columns=["month"])
        return train, test

    def inner_converter(inner_df):
        inner_df = inner_df.reset_index().melt(id_vars=["datetime"], var_name="id", value_name="usage")
        inner_df = inner_df.sort_values(["id", "datetime"])
        inner_df = inner_df.rename(columns={"datetime": "date"})
        inner_df = inner_df.set_index("date")
        return inner_df

    path = root + "datasets/air/pm25_ground.txt"
    real_df = pd.read_csv(path, parse_dates=['datetime'], date_parser=pd.to_datetime, index_col="datetime")
    path = root + "datasets/air/pm25_missing.txt"
    missing_df = pd.read_csv(path, parse_dates=['datetime'], date_parser=pd.to_datetime, index_col="datetime")
    print("datasets values  count = ", missing_df.shape[0] * missing_df.shape[1])
    print("both datasets are nan = ", (missing_df.isna() & real_df.isna()).sum().sum())
    print("only missing dataset is nan = ", (missing_df.isna() & ~real_df.isna()).sum().sum())
    # print((missing_df.isna() & ~real_df.isna()).sum())
    real_df = inner_converter(real_df)
    filled_df = real_df.groupby("id").apply(Interpolation.fill_nan, "linear")
    filled_real_df = pd.DataFrame()
    for row in filled_df:
        user_id = row[1]
        temp_df = row[0]
        temp_df["id"] = user_id
        filled_real_df = filled_real_df.append(temp_df)
    real_df = real_df.reset_index().set_index(["date", "id"])
    filled_real_df = filled_real_df.reset_index().rename(
        columns={"index": "date", "predicted_usage": "usage"}).set_index(["date", "id"])

    real_df.loc[real_df.index.isin(filled_real_df.index), "usage"] = filled_real_df.usage

    missing_df = inner_converter(missing_df)
    train_x, test_x = split_train_test(real_df)
    train_x_nan, test_x_nan = split_train_test(missing_df.reset_index().set_index(["date", "id"]))
    train_x.to_csv(root + "datasets/air/train_x.csv")
    test_x.to_csv(root + "datasets/air/test_x.csv")
    train_x_nan.to_csv(root + "datasets/air/train_x_nan.csv")
    test_x_nan.to_csv(root + "datasets/air/test_x_nan.csv")


if __name__ == '__main__':
    load_files_two()
