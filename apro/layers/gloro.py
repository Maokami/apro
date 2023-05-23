import tensorflow as tf

from gloro.layers import AveragePooling2D as GloroAveragePooling2D
from gloro.layers import Conv2D as GloroConv2D
from gloro.layers import Dense as GloroDense
from gloro.layers import Flatten as GloroFlatten
from gloro.layers import MaxPooling2D as GloroMaxPooling2D
from gloro.layers import ReLU as GloroReLU

from tensorflow.keras.layers import Lambda as KerasLambda

from apro.layers.base import AproLayer

import math


class Dense(GloroDense, AproLayer):
    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=1, keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    def bound(self, input_bound):
        lc = self.lipschitz_inf()
        lower_bound, upper_bound = input_bound
        return (lower_bound * lc, upper_bound * lc)


class Conv2D(GloroConv2D, AproLayer):
    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=[0, 1, 2], keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    # TODO: bias
    def bound(self, input_bound):
        lc = self.lipschitz_inf()
        lower_bound, upper_bound = input_bound
        return (lower_bound * lc, upper_bound * lc)


class AveragePooling2D(GloroAveragePooling2D, AproLayer):
    # TODO : input_shape
    def lipschitz_inf(self):
        w = (
            tf.eye(self.input.shape[-1])[None, None]
            * (tf.ones(self.pool_size)[:, :, None, None])
            / (self.pool_size[0] * self.pool_size[1])
        )

        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=[0, 1, 2], keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    def bound(self, input_bound):
        return input_bound


class Flatten(GloroFlatten, AproLayer):
    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        return input_bound


# TODO
class MaxPooling2D(GloroMaxPooling2D, AproLayer):
    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        return input_bound


class ReLU(GloroReLU, AproLayer):
    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_bound):
        _, upper_bound = input_bound
        return (0, upper_bound)


class ApproxReLU(KerasLambda, AproLayer):
    def __init__(self, alpha, B=1, **kwargs):
        self.alpha = alpha
        self.B = B
        self.range = (-B, B)

        super().__init__(self.relu_approx, **kwargs)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_B(self, B):
        self.B = B
        self.range = (-B, B)

    # TODO
    def relu_approx(self, x):
        return tf.where(x >= 0, x, self.alpha * x)

    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def approx_error(self):
        return 2 ** (-self.alpha) * self.B

    def propagate_error(self, error):
        return error + self.approx_error()

    def bound(self, input_bound):
        lower_bound, upper_bound = input_bound
        B = math.ceil(max(abs(lower_bound), abs(upper_bound)))
        self.set_B(B)
        return (0, upper_bound)
