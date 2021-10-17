from pathlib import Path

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_with_modified_date
from src.utils.parallelizem import apply_parallel


def read_data(address):
    return pd.read_csv(Path(address))


def select_user_data(df, id):
    return df.loc[df.id == id].copy()


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    return user, scaler


def preimputation(user: pd.DataFrame):
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    complete_row = user[~user["usage"].isna()]
    # Could do this for a percentage of data
    user['usage'] = user['usage'].interpolate().ffill().bfill()
    return user, nan_index


def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset.iloc[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back]['usage'])
    return np.array(dataX), np.array(dataY)


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
    model.fit(train, consumptions, epochs=150
              , batch_size=batch_size, verbose=2, shuffle=False)

    return model


def testing(model, user, nan_index, train, look_back):
    nan_index = nan_index - (user.index[0] + look_back + 1)
    nan_index = nan_index[nan_index >= 0]
    extra = train[nan_index[nan_index < 0] + look_back + 1]
    x_test = train[nan_index]
    # print(x_test)
    prediction = model.predict(x_test)
    return prediction, extra


def fill_nan(main_user):
    user = main_user.copy()
    # user = select_user_data(df, 2)
    user_preimputed, nan_index = preimputation(user)
    user, scaler = normalize_user_usage(user)
    look_back = 12
    x, consumptions = create_dataset(user_preimputed, look_back)
    model = training(x, consumptions, look_back)
    prediction, extra = testing(model, user_preimputed, nan_index, x, look_back)
    prediction = np.reshape(prediction, (prediction.shape[0]))
    extra = np.array(extra)
    extra = np.reshape(extra, (extra.shape[0]))
    prediction = np.concatenate((extra, prediction), axis=0)
    prediction = prediction.reshape(1, -1)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction.reshape(prediction.shape[1])
    return pd.Series([prediction, nan_index])


if __name__ == '__main__':
    # address = 'h:/Projects/Datasets/Smart/with_nan/smart_star_small_date_modified_0.05.csv'
    # address = 'E:/HandlingMissingValues/datasets/with_nan/smart_star_small_date_modified_0.01.csv'
    # df = read_data(address)
    x, x_nan = get_dataset_with_modified_date()
    x_nan.drop(columns=['year', 'winter', 'spring', 'summer', 'fall'], inplace=True)
    filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))
