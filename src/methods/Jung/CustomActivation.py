import tensorflow as tf


def mmd(x):
    return tf.nn.tanh(x) ** 2
