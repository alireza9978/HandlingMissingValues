import pandas
import pandas as pd
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def read_data(address):
    return pd.read_csv(Path(address))


def select_user_data(df, id):
    return df.loc[df.id == id].copy()


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user[['usage']] = scaler.fit_transform(user[['usage']])


def preimputation(user: pandas.DataFrame):
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    complete_row = user[~user["usage"].isna()]
    user['usage'] = user['usage'].ffill().bfill()
    return user, nan_index


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def model():
    pass


if __name__ == '__main__':
    address = 'h:/Projects/Datasets/Smart/with_nan/smart_star_small_date_modified_0.05.csv'
    df = read_data(address)
    user = select_user_data(df, 105)
    normalize_user_usage(user)
    user, nan_index = preimputation(user)
