import pandas
import pandas as pd
import numpy as np
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset


def read_data(address):
    return pd.read_csv(Path(address))


def select_user_data(df, id):
    return df.loc[df.id == id].copy()


def normalize_user_usage(user):
    scaler = MinMaxScaler()
    user[['usage']] = scaler.fit_transform(user[['usage']])
    return user, scaler


def preimputation(user: pandas.DataFrame):
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    complete_row = user[~user["usage"].isna()]
    # Could do this for a percentage of data
    user['usage'] = user['usage'].ffill().bfill()
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
    model.add(LSTM(10, input_shape=(look_back, train.shape[2]), stateful=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train, consumptions, epochs=10, batch_size=batch_size, verbose=2, shuffle=False)
    return model


def testing(model, user, nan_index, train, look_back):
    nan_index = nan_index - (user.index[0] + look_back + 1)
    nan_index = nan_index[nan_index >= 0]
    extra = train[nan_index[nan_index < 0] + look_back + 1]
    x_test = train[nan_index]
    # print(x_test)
    prediction = model.predict(x_test)
    return prediction, extra


def fill_nan(user):
    # user = select_user_data(df, 2)
    user, scaler = normalize_user_usage(user)
    user_preimputed, nan_index = preimputation(user)
    look_back = 10
    x, consumptions = create_dataset(user_preimputed, look_back)
    model = training(x, consumptions, look_back)
    prediction,extra = testing(model, user_preimputed, nan_index, x, look_back)
    prediction = np.reshape(prediction, (prediction.shape[0]))
    extra = np.array(extra)
    extra = np.reshape(extra, (extra.shape[0]))
    prediction = np.concatenate((extra,prediction),axis=0)
    prediction = scaler.inverse_transform(prediction)
    return pd.Series([prediction,nan_index])


if __name__ == '__main__':
    address = 'h:/Projects/Datasets/Smart/with_nan/smart_star_small_date_modified_0.05.csv'
    df = read_data(address)
    df.drop(columns=['year', 'winter', 'spring', 'summer', 'fall'], inplace=True)
    x, x_nan = get_dataset()
    filled_users = df.groupby("id").apply(fill_nan)
    filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    print(evaluate_dataframe(filled_users, mean_square_error))