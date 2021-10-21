import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.methods.Jung.SmallNeuralNet import SmallNeuralNet
from src.preprocessing.load_dataset import get_dataset_fully_modified_date, root


class MultiLayerPerceptron:

    def __init__(self, temp_df: pd.DataFrame, user_id):
        self.user_id = user_id
        self.small_neural_net_count = 20
        self.training_epoch = 50
        self.x_scaler = MultiLayerPerceptron._generate_scaler()
        self.y_scaler = MultiLayerPerceptron._generate_scaler()
        self._preprocess_data_set(temp_df)
        self.model = self._generate_model(self.small_neural_net_count)

    @staticmethod
    def _generate_scaler():
        return MinMaxScaler()

    def _preprocess_data_set(self, temp_df: pd.DataFrame):
        temp_df = temp_df[~temp_df.usage.isna()]
        y_columns = ["usage"]
        useless_columns = ["id"]

        train_x = temp_df.drop(columns=useless_columns + y_columns)
        temp_columns = train_x.columns
        train_x = train_x.to_numpy()
        train_y = temp_df[y_columns].to_numpy()

        train_x = self.x_scaler.fit_transform(train_x)
        train_y = self.y_scaler.fit_transform(train_y)

        self.dataset = pd.DataFrame(train_x, columns=temp_columns)
        self.dataset[y_columns] = self.y_scaler.fit_transform(train_y)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32)

        self._save_scaler()

    def _save_scaler(self):
        dump(self.x_scaler, root + f'saved_models/neural_net_scaler_x_{self.user_id}.joblib')
        dump(self.y_scaler, root + f'saved_models/neural_net_scaler_y_{self.user_id}.joblib')

    def _generate_model(self, input_shape):
        self.small_neural_nets = []
        for i in range(input_shape):
            self.small_neural_nets.append(SmallNeuralNet(self.dataset, self.x_scaler, self.y_scaler))

        inputs = layers.Input(shape=(input_shape,))
        output = layers.Dense(1, activation="softmax")(inputs)
        model = keras.Model(inputs, output)
        model.compile("adam", "mean_square_error")
        return model

    def train_model(self):
        maximum_loss = 0.6
        maximum_iteration = 10
        current_iteration = 1
        while True:
            for small_model in self.small_neural_nets:
                small_model.train_model()

            self.model.fit(self.train_dataset, epochs=self.training_epoch)
            self.model.save(root + f'saved_models/neural_net_{self.user_id}.h5')
            temp_loss = 1

            if temp_loss < maximum_loss:
                print("converged")
                break
            if current_iteration >= maximum_iteration:
                print("did not converge")
                break


if __name__ == '__main__':
    x, x_nan = get_dataset_fully_modified_date()
    x = x[x.id == 45]
    x_nan = x_nan[x_nan.id == 45]
    temp = MultiLayerPerceptron(x_nan, 45)
