import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Conv1D, Conv1DTranspose
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

from src.measurements.Measurements import mean_square_error, evaluate_dataframe_two
from src.preprocessing.smart_star.load_dataset import get_dataset_fully_modified_date_auto


class DAE:

    def __init__(self, temp_df: pd.DataFrame):
        self.df = temp_df
        self.train_percent = 0.3
        self.vector_length = 24
        self.batch_size = 32
        self.epochs = 200

        self.autoencoder = None
        self.train_df = None
        self.test_df = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.df_nan_indexes = None
        self.scaler = DAE._generate_scaler()
        self.preimputation()
        self.normalize_user_usage()
        self.split_df()
        self.preparation()

    @staticmethod
    def _generate_scaler():
        return MinMaxScaler()

    def split_df(self):
        total_count = self.df.shape[0]
        starting_index = self.df.index[0]
        ending_index = self.df.index[-1]
        test_start_index = ending_index - int(total_count * self.train_percent)
        self.df.loc[starting_index:test_start_index - 1, "train"] = True
        self.df.loc[test_start_index:ending_index, "train"] = False
        self.train_df = self.df[self.df.train].copy().drop(columns=["train"]).reset_index(drop=True)
        self.test_df = self.df[self.df.train == False].copy().drop(columns=["train"]).reset_index(drop=True)
        self.df_nan_indexes = self.df_nan_indexes[self.df.train == False].index

    def normalize_user_usage(self):
        self.df['usage'] = self.scaler.fit_transform(self.df['usage'].to_numpy().reshape(-1, 1))
        self.df['real_usage'] = self.scaler.transform(self.df['real_usage'].to_numpy().reshape(-1, 1))

    def preimputation(self):
        self.df_nan_indexes = self.df["usage"].isna()
        self.df['usage'] = self.df['usage'].fillna(-1)

    def preparation(self):
        self.train_df = self.train_df.drop(columns=['id'])
        self.test_df = self.test_df.drop(columns=['id'])

        user = self.train_df.drop(columns=['real_usage'])

        real = self.train_df.copy()
        real["usage"] = real["real_usage"]
        real = real.drop(columns=['real_usage'])

        user_test = self.test_df.drop(columns=['real_usage'])

        real_test = self.test_df.copy()
        real_test["usage"] = real_test["real_usage"]
        real_test = real_test.drop(columns=['real_usage'])

        result = []
        for temp in [user, real, user_test, real_test]:
            temp = temp.to_numpy()
            temp = temp[:int(temp.shape[0] / self.vector_length) * self.vector_length, :]
            temp = temp.reshape(int(temp.shape[0] / self.vector_length), self.vector_length, temp.shape[1])
            result.append(temp)
        self.train_x = result[0]
        self.train_y = result[1]
        self.test_x = result[2]
        self.test_y = result[3]

    def train(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).batch(self.batch_size)
        input_layer = Input(shape=(self.train_x.shape[1], self.train_x.shape[2]))
        # input_shape = (batch_size,train.shape[1],train.shape[2])

        # Encoder
        layer1 = Conv1D(64, 2, padding='same', name='layer1-conv1d')(input_layer)
        layer1 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-relu')(layer1)
        # layer2 = Dense(32)(layer1)
        layer2 = Conv1D(64, 2, padding='same', name='layer2-conv1d')(layer1)
        layer2 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-relu')(layer2)
        layer3 = Conv1D(64, 2, padding='same', name='layer3-conv1d')(layer2)
        layer3 = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer3-relu')(layer3)
        encodings = Dense(16, name='encodings')(layer3)
        # Decoder
        # layer2_ = Dense(32)(encodings)
        layer3_ = Conv1DTranspose(64, 2, padding='same', name='layer3-conv1dTranspose')(encodings)
        layer3_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer3-reluT')(layer3_)
        layer2_ = Conv1DTranspose(32, 2, padding='same', name='layer2-conv1dTranspose')(layer3_)
        layer2_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer2-reluT')(layer2_)
        layer1_ = Conv1DTranspose(1, 2, padding='same', name='layer1-conv1dTranspose')(layer2_)
        layer1_ = tf.keras.layers.LeakyReLU(alpha=0.3, name='layer1-reluT')(layer1_)
        decoded = Conv1D(self.train_x.shape[2], 3, activation="sigmoid", padding="same", name='decodings')(layer1_)
        autoencoder = Model(input_layer, decoded)
        # optimizer = tf.optimizers.Adam(clipvalue=0.5)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        print(autoencoder.summary())
        autoencoder.fit(train_dataset, epochs=self.epochs, verbose=2)
        # model.fit(train, consumptions, epochs=150, batch_size=batch_size, verbose=2, shuffle=False)
        self.autoencoder = autoencoder

    def test(self):
        test_dataset = tf.data.Dataset.from_tensor_slices(self.test_x).batch(self.batch_size)
        predictions = self.autoencoder.predict(test_dataset)
        return predictions


def fill_nan(temp_df: pd.DataFrame):
    model = DAE(temp_df)
    model.train()
    predictions = model.test()

    test_set_user = model.test_x[:, :, 0].reshape(model.test_x.shape[0] * model.test_x.shape[1])
    test_set_user = model.scaler.inverse_transform(test_set_user.reshape(-1, 1)).reshape(test_set_user.shape[0], )
    predictions = predictions[:, :, 0]  # usage is in the second column
    predictions = predictions.reshape(predictions.shape[0] * predictions.shape[1])
    predictions = model.scaler.inverse_transform(predictions.reshape(-1, 1))
    nan_indices = np.where(test_set_user == -1)[0]
    real = model.test_y[:, :, 0]  # usage is in the second column
    real = real.reshape(real.shape[0] * real.shape[1])
    real = model.scaler.inverse_transform(real.reshape(-1, 1))
    # print(real)
    return pd.DataFrame(
        {'usage': temp_df.loc[model.df_nan_indexes.to_numpy().squeeze()[-1 * nan_indices.shape[0]:]][
            'usage'].to_numpy(),
         "predicted_usage": predictions.reshape(predictions.shape[0] * predictions.shape[1])[nan_indices]},
        index=model.df_nan_indexes.to_numpy().squeeze()[-1 * nan_indices.shape[0]:])


if __name__ == '__main__':
    main_df = get_dataset_fully_modified_date_auto("0.05")
    main_df = main_df[main_df.id == 99]
    main_df.drop(columns=['year', 'winter', 'spring', 'summer', 'fall', 'holiday', 'weekend', 'temperature',
                          'humidity', 'visibility', 'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover',
                          'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability'], inplace=True)
    t = []
    # for i in range(20):
    filled_users = main_df.groupby("id").apply(fill_nan)

    error, error_df = evaluate_dataframe_two(filled_users, mean_square_error)
    # print(error)
    # print(error_df)
    t.append(error)
    print(t)
    # print(user)
    # filled_users = apply_parallel(x_nan.groupby("id"), fill_nan)
    # # filled_users = x_nan.groupby("id").apply(fill_nan)
    # filled_users[2] = filled_users[1].apply(lambda idx: x.loc[idx])
    # print(evaluate_dataframe(filled_users, mean_square_error))
