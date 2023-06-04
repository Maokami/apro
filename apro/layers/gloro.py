import tensorflow as tf

from gloro.layers import AveragePooling2D as GloroAveragePooling2D
from gloro.layers import Conv2D as GloroConv2D
from gloro.layers import Dense as GloroDense
from gloro.layers import Flatten as GloroFlatten
from gloro.layers import MaxPooling2D as GloroMaxPooling2D
from gloro.layers import ReLU as GloroReLU

from tensorflow.keras.layers import Lambda as KerasLambda

from apro.layers.base import AproLayer
from apro.approximation import ReLU_approx, rangeException

import math
import numpy as np

import warnings


class Dense(GloroDense, AproLayer):
    def __init__(self, *args, B=None, **kwargs):
        self.B = B
        super().__init__(*args, **kwargs)

    def set_B(self, B):
        self.B = B

    def call(self, inputs):
        max_val = tf.norm(inputs, ord=np.inf)
        if self.B is not None:
            condition = tf.reduce_sum(tf.cast(tf.abs(inputs) > self.B, tf.int32)) != 0
            tf.print(
                tf.where(
                    condition,
                    f"Dense: max_val ({max_val}) exceeds B ({self.B})",
                    "",
                )
            )
        return super().call(inputs)

    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=0, keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    def bound(self, input_B):
        lc = self.lipschitz_inf()
        bias = self.get_weights()[1]
        b = tf.reduce_max(abs(bias))
        return input_B * lc + b


class Conv2D(GloroConv2D, AproLayer):
    def __init__(self, *args, B=None, **kwargs):
        self.B = B
        super().__init__(*args, **kwargs)

    def set_B(self, B):
        self.B = B

    def call(self, inputs):
        max_val = tf.norm(inputs, ord=np.inf)
        if self.B is not None:
            condition = tf.reduce_sum(tf.cast(tf.abs(inputs) > self.B, tf.int32)) != 0
            tf.print(
                tf.where(
                    condition,
                    f"Conv: max_val ({max_val}) exceeds B ({self.B})",
                    "",
                )
            )
        return super().call(inputs)

    def lipschitz_inf(self):
        w = self.kernel
        lc = tf.reduce_max(tf.reduce_sum(tf.abs(w), axis=[0, 1, 2], keepdims=False))
        return lc

    def propagate_error(self, error):
        return self.lipschitz_inf() * error

    # TODO: bias
    def bound(self, input_B):
        lc = self.lipschitz_inf()
        bias = self.get_weights()[1]
        b = tf.reduce_max(abs(bias))
        return input_B * lc + b


class AveragePooling2D(GloroAveragePooling2D, AproLayer):
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

    def bound(self, input_B):
        return input_B


class Flatten(GloroFlatten, AproLayer):
    def __init__(self, *args, B=None, **kwargs):
        self.B = B
        super().__init__(*args, **kwargs)

    def set_B(self, B):
        self.B = B

    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_B):
        return input_B


# TODO
class MaxPooling2D(GloroMaxPooling2D, AproLayer):
    def lipschitz_inf(self):
        return 1.0

    def propagate_error(self, error):
        return error

    def bound(self, input_B):
        return input_B


class ReLU(GloroReLU, AproLayer):
    def __init__(self, alpha, B=1, **kwargs):
        self.alpha = alpha
        self.B = B
        self.range = (-B, B)

        super().__init__(**kwargs)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_B(self, B):
        self.B = B
        self.range = (-B, B)

    def call(self, inputs):
        max_val = tf.norm(inputs, ord=np.inf)
        return super().call(inputs)

    def lipschitz_inf(self):
        return 1.0

    def approx_error(self):
        return 2 ** (-self.alpha) * self.B

    def propagate_error(self, error):
        return error + self.approx_error()

    def bound(self, input_B):
        self.set_B(input_B)
        return input_B + self.approx_error()


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
        relu_dict = {"alpha": self.alpha, "B": self.B}
        return ReLU_approx(x, relu_dict)

    def lipschitz(self):
        return 1.0

    def lipschitz_inf(self):
        return 1.0

    def approx_error(self):
        return 2 ** (-self.alpha) * self.B

    def propagate_error(self, error):
        return error + self.approx_error()

    def bound(self, input_B):
        return input_B + self.approx_error()
