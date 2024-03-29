import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing.smart_star.load_dataset import get_dataset_fully_modified_date, root


class SmallNeuralNet:
    id = 1

    def __init__(self, temp_df: pd.DataFrame, x_scaler, y_scaler, training_epochs):
        self.id = SmallNeuralNet.id
        SmallNeuralNet.id += 1
        self.history = None
        self.training_epochs = training_epochs
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.input_shape = self._preprocess_data_set(temp_df)
        self.model = SmallNeuralNet._generate_model(self.input_shape)

    def _preprocess_data_set(self, temp_df: pd.DataFrame):
        temp_df = temp_df[~temp_df.usage.isna()]
        indexes = temp_df.index.to_numpy()
        selected_indexes = np.random.choice(indexes, int(indexes.shape[0] * 0.3))
        temp_df = temp_df.loc[selected_indexes]

        y_columns = ["usage"]

        train_y = temp_df[y_columns].to_numpy()
        train_x = temp_df.drop(columns=y_columns).to_numpy()

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)

        return train_x.shape[1]

    @staticmethod
    def _generate_model(input_shape):
        inputs = layers.Input(shape=(input_shape,))
        model = layers.Dense(32, activation="relu")(inputs)
        model = layers.Dropout(0.25)(model)
        model = layers.Dense(64, activation="relu")(model)
        model = layers.Dropout(0.25)(model)
        model = layers.Dense(16, activation="relu")(model)
        model = layers.Dropout(0.25)(model)
        output = layers.Dense(1, activation="sigmoid")(model)
        model = keras.Model(inputs, output)
        model.compile("adam", "mean_squared_error")
        return model

    def train_model(self):
        self.history = self.model.fit(self.train_dataset, epochs=self.training_epochs)
        self.model.save(root + f'saved_models/neural_net_{self.id}.h5')

    def save_plot(self):
        if self.history is not None:
            plt.plot(self.history.history['loss'], label='train_loss')
            plt.legend()
            plt.savefig(root + f"results/Jung/small_neural_net/{self.id}.jpeg")
            plt.close()

    def predict(self, test_x: np.array):
        if test_x.shape[1] == self.input_shape:
            return self.model.predict(test_x)
        print("bad input")
        return None


if __name__ == '__main__':
    temp_x_scaler = MinMaxScaler()
    temp_y_scaler = MinMaxScaler()
    x, x_nan = get_dataset_fully_modified_date()
    x_nan = x_nan[x_nan.id == 45]
    x_nan = x_nan[~x_nan.usage.isna()]
    temp_y_columns = ["usage"]
    useless_columns = ["id"]

    temp_train_x = x_nan.drop(columns=useless_columns + temp_y_columns)
    temp_columns = temp_train_x.columns
    temp_train_x = temp_train_x.to_numpy()
    temp_train_y = x_nan[temp_y_columns].to_numpy()

    temp_train_x = temp_x_scaler.fit_transform(temp_train_x)
    temp_train_y = temp_y_scaler.fit_transform(temp_train_y)
    dataset = pd.DataFrame(temp_train_x, columns=temp_columns)
    dataset[temp_y_columns] = temp_y_scaler.fit_transform(temp_train_y)

    temp_model = SmallNeuralNet(dataset, temp_x_scaler, temp_y_scaler)
    temp_model.training_epochs = 20
    temp_model.train_model()
    temp_model.save_plot()
