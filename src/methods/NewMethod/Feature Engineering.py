from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, Conv1DTranspose
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.smart_star.load_dataset import get_dataset
from src.preprocessing.smart_star.load_dataset import root
from src.utils.Dataset import get_user_by_id


def calculate_feature(temp_df: pd.DataFrame, window_size: int):
    import swifter
    _ = swifter.config

    minimum_index = temp_df.index[0]
    maximum_index = temp_df.index[-1]

    nan_indexes = temp_df.usage.isna()
    result_df = pd.DataFrame()
    for temp_index in nan_indexes[nan_indexes].index:
        temp_minimum_index = max((temp_index - window_size, minimum_index))
        temp_maximum_index = min((temp_index + window_size, maximum_index))
        left_feature = temp_df.loc[temp_minimum_index:temp_index - 1].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        left_feature.index += "_left"
        right_feature = temp_df.loc[temp_index + 1:temp_maximum_index].usage.agg(
            {"usage_sum": "sum", "usage_min": "min", "usage_max": "max", "usage_mean": "mean",
             "usage_median": "median", "usage_var": "var", "usage_std": "std", "usage_skew": "skew",
             "usage_kurt": "kurt", "usage_count": "count"})
        right_feature.index += "_right"
        feature_row = pd.concat([left_feature, right_feature])
        feature_row.name = temp_index
        result_df = result_df.append(feature_row)

    return result_df


def data_preparation(train_x, columns):
    train = train_x.copy()
    for column in columns:
        scaler = MinMaxScaler()
        train[column] = scaler.fit_transform(train[column].to_numpy().reshape(-1, 1))
    train = train.to_numpy().reshape(train.shape[0], 1, train.shape[1])
    return train


def preimputation(train_x, column):
    train_x[column] = train_x[column].fillna(method='bfill').fillna(method='ffill')
    return train_x


def create_neighborhood(train, window_size):
    new_dataset = []
    for i in range(window_size, train.shape[0] - window_size):
        new_dataset.append(train[i - window_size:i + window_size + 1])
    return np.array(new_dataset)


def feature_extractor(train_x, mode):
    batch_size = 32
    epochs = 150
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_x)).batch(batch_size)
    input_layer = Input(shape=(train_x.shape[1], train_x.shape[2]))

    # Encoder
    if mode == 1:
        layer1 = Dense(16)(input_layer)
    elif mode == 2:
        layer1 = Conv1D(16, 3, padding='same')(input_layer)
    elif mode == 3:
        layer1 = Conv1D(16, 3, padding='valid')(input_layer)
    layer1 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-relu')(layer1)
    layer2 = Dense(8)(layer1)
    layer2 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-relu')(layer2)
    layer2 = tf.keras.layers.Dropout(0.2)(layer2)
    encodings = Dense(5, name='encodings')(layer2)
    # Decoder

    layer2_ = Dense(8)(encodings)
    layer2_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-reluT')(layer2_)
    layer2_ = tf.keras.layers.Dropout(0.2)(layer2_)
    if mode == 1:
        layer1_ = Dense(16)(layer2_)
    if mode == 2:
        layer1_ = Conv1DTranspose(16, 3, padding='same')(layer2_)
    if mode == 3:
        layer1_ = Conv1D(16, 3, padding='valid')(layer2_)
    layer1_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-reluT')(layer1_)
    decoded = Dense(train_x.shape[2], activation="sigmoid", name='decodings')(layer1_)
    autoencoder = Model(input_layer, decoded)
    # optimizer = tf.optimizers.Adam(clipvalue=0.5)
    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    print(autoencoder.summary())
    autoencoder.fit(train_dataset, epochs=epochs, verbose=2)
    # autoencoder.fit(train_x, train_x, validation_split=0.30, batch_size=batch_size, epochs=epochs, verbose=2)
    encoder = Model(input_layer, encodings)
    return encoder.predict(train_x)


# using all only the moving features of the users in feature extraction
def feature_extraction_moving_features(t_nan):
    moving_features = t_nan.groupby("id").apply(calculate_feature, 12)
    train_x = moving_features.dropna()
    nan_indexes = train_x.index
    encodings = feature_extractor(data_preparation(train_x, train_x.columns), 1)
    return encodings, nan_indexes


# using only the original features of the users in feature extraction
def feature_extraction_original_data(t_nan):
    window_size = 12
    train_x = t_nan.copy()
    nan_indices = train_x[train_x.usage.isna()].index.to_numpy()
    train_x = preimputation(train_x, "usage")
    scaler = MinMaxScaler()
    train_x['usage'] = scaler.fit_transform(train_x['usage'].to_numpy().reshape(-1, 1))
    usage = train_x['usage'].values
    usage = np.pad(usage, (window_size, window_size), 'constant', constant_values=(usage[0], usage[-1]))
    usage = create_neighborhood(usage, window_size)
    usage = usage[nan_indices]
    usage = usage.reshape(usage.shape[0], 1, usage.shape[1])
    encodings = feature_extractor(usage, 2)
    return encodings


# using all the features of the users in feature extraction
def feature_extraction_combination(train_x, ):
    feature_extractor(data_preparation(train_x, train_x.columns), 3)


if __name__ == '__main__':
    nan_percent = "0.01"
    x, x_nan = get_dataset(nan_percent)
    for user_id in [99, 12, 65, 35]:
        (user_x, user_x_nan) = get_user_by_id(x, x_nan, user_id)
        moving_features_encodings, result_index = feature_extraction_moving_features(user_x_nan)
        moving_features_encodings_df = pd.DataFrame(
            moving_features_encodings.reshape(moving_features_encodings.shape[0], moving_features_encodings.shape[2]),
            index=result_index)
        target_path = Path(root + "datasets/extracted_features/encodings_moving_features_" +
                           str(user_id) + "_" + nan_percent + ".csv")
        moving_features_encodings_df.to_csv(target_path)

    # user_encodings = feature_extraction_original_data(x_nan)
    # pd.DataFrame(user_encodings.reshape(user_encodings.shape[0], user_encodings.shape[2])).to_csv(
    #     Path(root + "encodings_original_data_" + str(user_id) + "_" + nan_percent + ".csv"))
    # feature_extraction_original_data(t_nan)
