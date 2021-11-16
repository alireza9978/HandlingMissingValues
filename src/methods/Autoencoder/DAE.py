from pathlib import Path

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def read_data(address):
    return pd.read_csv(Path(address))


def select_user_data(df, temp_id):
    return df.loc[df.id == temp_id].copy()


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    return user, scaler


def preimputation(user: pd.DataFrame):
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    # Could do this for a percentage of data
    user['usage'] = user['usage'].fillna(0)
    return user, nan_index

def training(train, consumptions, look_back):
    # trainX = np.reshape(train, (train.shape[0], train.shape[1], 1))
    # print(trainX[0])
    # create and fit the LSTM network
    batch_size = 32
    model = Sequential()
    model.add(LSTM(64, input_shape=(look_back, train.shape[2]), stateful=False, return_sequences=False))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adadelta')
    model.fit(train, consumptions, epochs=150, batch_size=batch_size, verbose=2, shuffle=False)
    return model


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.05")
    x_nan.drop(columns=['year', 'winter', 'spring', 'summer', 'fall', 'holiday', 'weekend', 'temperature', 'humidity',
                        'visibility', 'apparentTemperature', 'pressure', 'windSpeed',
                        'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint',
                        'precipProbability'], inplace=True)
    user = select_user_data(x_nan, 100)
    print(user.shape)
    print(user.head())
    # filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    # # filled_users = x_nan.groupby("id").apply(fill_nan)
    # filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    # print(evaluate_dataframe(filled_users, mean_square_error))