import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers

from src.methods.Jung.CustomActivation import mmd
from src.methods.Jung.CustomWeightConstraints import KernelConstraint, BiasConstraint

# y_pred = model.predict(train_dataset.take(1))
#     print(y_pred)
#     w = np.power(math.e, train_x)
#     w = w / w.sum()
#     print(model.weights)

if __name__ == '__main__':
    input_shape = 10

    mlp_values = np.array([1, 2, 1, 4, 1, 6, 7, 91, 2.5, 4.66]).reshape(-1, input_shape)
    mlp_result = np.array([1]).reshape(-1, 1)
    mlp_values_dataset = tf.data.Dataset.from_tensor_slices((mlp_values, mlp_result)).batch(1)

    train_x = np.ones(10).reshape(-1, input_shape)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_x).batch(1)
    model = Sequential()
    # model.add(layers.Dense(500, input_dim=input_shape, activation='relu',
    #                        kernel_constraint=min_max_norm(min_value=1.0, max_value=1.0)))
    model.add(layers.Dense(1, input_dim=input_shape, activation=layers.Activation(mmd),
                           kernel_constraint=KernelConstraint(),
                           bias_constraint=BiasConstraint()))
    model.compile("adam", "mean_squared_error")
    model.fit(mlp_values_dataset, epochs=10)
    print(model.get_weights())
