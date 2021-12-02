import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, Conv1DTranspose
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing.load_dataset import get_dataset
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


def data_preparation(train_x):
    train = train_x.copy()
    for column in train.columns:
        scaler = MinMaxScaler()
        train[column] = scaler.fit_transform(train[column].to_numpy().reshape(-1, 1))
    train = train.to_numpy().reshape(train.shape[0], 1, train.shape[1])
    return train


def preimputation():
    pass


def feature_extractor(train_x, mode):
    batch_size = 32
    epochs = 100
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_x)).batch(batch_size)
    input_layer = Input(shape=(train_x.shape[1], train_x.shape[2]))

    # Encoder
    if mode == 1:
        layer1 = Dense(16)(input_layer)
    elif mode == 2:
        layer1 = Conv1D(16, 3, padding='valid')(input_layer)
    elif mode == 3:
        layer1 = Conv1D(16, 3, padding='valid')(input_layer)
    layer1 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-relu')(layer1)
    layer2 = Dense(8)(layer1)
    layer2 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-relu')(layer2)
    encodings = Dense(5, name='encodings')(layer2)
    # Decoder
    layer2_ = Dense(8)(encodings)
    layer2_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-reluT')(layer2_)
    if mode == 1:
        layer1_ = Dense(16)
    if mode == 2:
        layer1_ = Conv1DTranspose(16, 3, padding='valid')(layer2_)
    if mode == 3:
        layer1_ = Conv1D(16, 3, padding='valid')(layer2_)
    layer1_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-reluT')(layer1_)
    decoded = Dense(train_x.shape[2], activation="sigmoid", name='decodings')(layer1_)
    autoencoder = Model(input_layer, decoded)
    # optimizer = tf.optimizers.Adam(clipvalue=0.5)
    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    print(autoencoder.summary())
    autoencoder.fit(train_dataset, epochs=epochs, verbose=2)
    encoder = Model(input_layer, encodings)
    return encoder.predict(train_x)


# using all only the moving features of the users in feature extraction
def feature_extraction_no_original(t_nan):
    moving_features = t_nan.groupby("id").apply(calculate_feature, 12)
    train_x = moving_features.dropna()
    encodings = feature_extractor(data_preparation(train_x), 1)


# using only the original features of the users in feature extraction
def feature_extraction_original(train_x):
    feature_extractor(data_preparation(train_x), 2)


# using all the features of the users in feature extraction
def feature_extraction_combination(train_x, ):
    feature_extractor(data_preparation(train_x), 3)


if __name__ == '__main__':
    nan_percent = "0.1"
    x, x_nan = get_dataset(nan_percent)
    for id in range(1, 2):
        (t, t_nan) = get_user_by_id(x, x_nan, id)
        feature_extraction_no_original(t_nan)
