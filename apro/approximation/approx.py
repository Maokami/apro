# Source : https://github.com/snu-ccl/approxCNN/blob/main/models/utils_approx.py

import tensorflow as tf
import numpy as np


class rangeException(Exception):
    def __init__(self, type, val):
        self.type = type
        self.val = val

    def show(self):
        if self.type == "relu":
            print(
                "STOP! There is an input value",
                self.val.item(),
                "for the approximate ReLU function.",
            )
        elif self.type == "max":
            print(
                "STOP! There is an input value",
                self.val.item(),
                "for the approximate max-pooling function.",
            )


def poly_eval(x, coeff):
    coeff = tf.constant(coeff, dtype=tf.float32)
    if len(x.shape) == 2:
        range_tensor = tf.range(tf.shape(coeff)[0], dtype=tf.float32)[None, None, :]
        x_expanded = tf.expand_dims(x, -1)
        result = x_expanded**range_tensor * coeff
        return tf.reduce_sum(result, axis=-1)
    elif len(x.shape) == 4:
        range_tensor = tf.range(tf.shape(coeff)[0], dtype=tf.float32)[
            None, None, None, None, :
        ]
        x_expanded = tf.expand_dims(x, -1)
        result = x_expanded**range_tensor * coeff
        return tf.reduce_sum(result, axis=-1)


def sgn_approx(x, relu_dict):
    alpha = relu_dict["alpha"]
    B = relu_dict["B"]

    # Get degrees
    f = open("./degreeResult/deg_" + str(alpha) + ".txt")
    readed = f.readlines()
    comp_deg = [int(i) for i in readed]

    # Get coefficients
    f = open("./coeffResult/coeff_" + str(alpha) + ".txt")
    coeffs_all_str = f.readlines()
    coeffs_all = [np.float32(i) for i in coeffs_all_str]
    i = 0

    if tf.reduce_sum(tf.cast(tf.abs(x) > B, tf.int32)) != 0:
        max_val = tf.reduce_max(tf.abs(x))
        raise rangeException("relu", max_val)

    x = x / B

    for deg in comp_deg:
        coeffs_part = coeffs_all[i : (i + deg + 1)]
        x = poly_eval(x, coeffs_part)
        i += deg + 1

    return x


def ReLU_approx(x, relu_dict):
    sgnx = sgn_approx(x, relu_dict)
    return x * (tf.constant(1.0, dtype=tf.float32) + sgnx) / 2
