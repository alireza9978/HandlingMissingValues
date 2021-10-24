import tensorflow as tf
import keras.backend as k
import numpy as np


class KernelConstraint(tf.keras.constraints.Constraint):

    def __init__(self):
        pass

    def __call__(self, w):
        w = k.pow(np.e, w)
        w = w / k.sum(w)
        return w


class BiasConstraint(tf.keras.constraints.Constraint):

    def __init__(self):
        pass

    def __call__(self, w):
        return w - w
