import tensorflow as tf
import keras.backend as k
import numpy as np


class KernelConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self):
        pass

    def __call__(self, w):
        w = k.pow(np.e, w)
        w = w / k.sum(w)
        return w

    def get_config(self):
        return {'ref_value': self.ref_value}


class BiasConstraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self):
        pass

    def __call__(self, w):
        return w - w

    def get_config(self):
        return {'ref_value': self.ref_value}
