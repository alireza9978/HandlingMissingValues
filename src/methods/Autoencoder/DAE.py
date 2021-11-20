from pathlib import Path

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, Conv1DTranspose
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from src.measurements.Measurements import evaluate_dataframe, mean_square_error
from src.preprocessing.load_dataset import get_dataset_fully_modified_date
from src.utils.parallelizem import apply_parallel


def read_data(address):
    return pd.read_csv(Path(address))


def select_user_data(df, temp_id):
    return df.loc[df.id == temp_id].copy()


def normalize_user_usage(user, real_data):
    scaler = MinMaxScaler()
    user['usage'] = scaler.fit_transform(user['usage'].to_numpy().reshape(-1, 1))
    real_data['usage'] = scaler.transform(real_data['usage'].to_numpy().reshape(-1, 1))
    return user, real_data, scaler


def preimputation(user: pd.DataFrame):
    nan_row = user[user["usage"].isna()]
    nan_index = nan_row.index.to_numpy()
    # Could do this for a percentage of data
    user['usage'] = user['usage'].fillna(-1)
    return user, nan_index


def prepration(user_data, real_data, vector_length):
    user = user_data.drop(columns=['id'])
    real = real_data.drop(columns=['id'])
    user = user.to_numpy()
    user = user[:int(user.shape[0] / vector_length) * vector_length, :]
    user = user.reshape(int(user.shape[0] / vector_length), vector_length, user.shape[1])
    real = real.to_numpy()
    real = real[:int(real.shape[0] / vector_length) * vector_length, :]
    real = real.reshape(int(real.shape[0] / vector_length), vector_length, real.shape[1])
    return user, real


def training(train, train_not_nan):
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((train, train_not_nan)).batch(32)
    input_layer = Input(shape=(train.shape[1], train.shape[2],))
    # input_shape = (batch_size,train.shape[1],train.shape[2])
    # Encoder
    layer1 = Conv1D(64, 3, padding='valid')(input_layer)
    layer2 = Dense(32, activation='relu')(layer1)
    encodings = Dense(16, activation='relu')(layer2)
    # Decoder
    layer2_ = Dense(32, activation='relu')(encodings)
    layer1_ = Conv1DTranspose(64, 3, padding='valid')(layer2_)
    decoded = Conv1D(train.shape[2], 3, activation="sigmoid", padding="same")(layer1_)
    autoencoder = Model(input_layer, decoded)
    # optimizer = tf.optimizers.Adam(clipvalue=0.5)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print(autoencoder.summary())
    autoencoder.fit(train_dataset, epochs=100, verbose=1)
    # model.fit(train, consumptions, epochs=150, batch_size=batch_size, verbose=2, shuffle=False)
    return autoencoder


def testing(test, autoencoder):
    test_dataset = tf.data.Dataset.from_tensor_slices(test).batch(32)
    predictions = autoencoder.predict(test_dataset)
    return predictions

def fill_nan(x,x_nan):
    user = select_user_data(x_nan, 100)
    real = select_user_data(x, 100)
    user, nan_index = preimputation(user)
    print(user.columns)
    user, real, scaler = normalize_user_usage(user, real)
    vector_length = 24
    user, real = prepration(user, real, vector_length)
    train_set_user = user[:500, :, :]
    train_set_real = real[:500, :, :]
    test_set_user = user[500:, :, :]
    test_set_real = real[500:, :, :]
    autoencoder = training(train_set_user, train_set_real)
    predictions = testing(test_set_user, autoencoder)
    predictions = scaler.inverse_transform(predictions)
    test_set_real = scaler.inverse_transform(test_set_real)
    print(predictions)



# train_dataset.take(1).as_numpy_iterator().next()[1]
if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date("0.05")
    x_nan.drop(columns=['year', 'winter', 'spring', 'summer', 'fall', 'holiday', 'weekend', 'temperature', 'humidity',
                        'visibility', 'apparentTemperature', 'pressure', 'windSpeed',
                        'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint',
                        'precipProbability'], inplace=True)
    x.drop(columns=['year', 'winter', 'spring', 'summer', 'fall', 'holiday', 'weekend', 'temperature', 'humidity',
                    'visibility', 'apparentTemperature', 'pressure', 'windSpeed',
                    'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint',
                    'precipProbability'], inplace=True)
    fill_nan(x,x_nan)
    # print(user)
    # filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    # # filled_users = x_nan.groupby("id").apply(fill_nan)
    # filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    # print(evaluate_dataframe(filled_users, mean_square_error))
