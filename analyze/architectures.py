from apro.layers import AveragePooling2D
from apro.layers import Conv2D
from apro.layers import Dense
from apro.layers import Flatten
from apro.layers import Input
from apro.layers import ReLU
from apro.layers import MaxPooling2D
from apro.layers import ApproxReLU

from tensorflow.keras.constraints import unit_norm


def cnn_2C2F(
    input_shape,
    num_classes,
    a,
    activation="relu",
    initialization="orthogonal",
):
    x = Input(input_shape)
    z = Conv2D(
        16,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(x)
    z = ReLU(a)(z)

    z = Conv2D(
        32,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = ReLU(a)(z)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initialization)(z)
    z = ReLU(a)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def approx_cnn_2C2F(
    input_shape,
    num_classes,
    initialization="orthogonal",
    a=7,
    B_list=[],
):
    x = Input(input_shape)
    z = Conv2D(
        16,
        4,
        strides=2,
        padding="same",
        B=B_list[0],
        kernel_initializer=initialization,
    )(x)
    z = ApproxReLU(a, B_list[1])(z)

    z = Conv2D(
        32,
        4,
        strides=2,
        padding="same",
        B=B_list[2],
        kernel_initializer=initialization,
    )(z)
    z = ApproxReLU(a, B_list[3])(z)

    z = Flatten()(z)
    z = Dense(100, B=B_list[5], kernel_initializer=initialization)(z)
    z = ApproxReLU(a, B_list[6])(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_4C3F(
    input_shape,
    num_classes,
    a,
    pooling="conv",
    activation="relu",
    initialization="orthogonal",
):
    x = Input(input_shape)

    z = Conv2D(32, 3, padding="same", kernel_initializer=initialization)(x)
    z = ReLU(a)(z)
    z = Conv2D(
        32,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = ReLU(a)(z)

    z = Conv2D(64, 3, padding="same", kernel_initializer=initialization)(z)
    z = ReLU(a)(z)
    z = Conv2D(
        64,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = ReLU(a)(z)

    z = Flatten()(z)
    z = Dense(
        512, kernel_initializer=initialization, kernel_constraint=unit_norm(axis=1)
    )(z)
    z = ReLU(a)(z)
    z = Dense(
        512, kernel_initializer=initialization, kernel_constraint=unit_norm(axis=1)
    )(z)
    z = ReLU(a)(z)

    y = Dense(
        num_classes,
        kernel_initializer=initialization,
        kernel_constraint=unit_norm(axis=1),
    )(z)

    return x, y


def approx_cnn_4C3F(
    input_shape,
    num_classes,
    initialization="orthogonal",
    a=7,
    B_list=[],
):
    x = Input(input_shape)

    z = Conv2D(32, 3, B=B_list[0], padding="same", kernel_initializer=initialization)(x)
    z = ApproxReLU(a, B_list[1])(z)
    z = Conv2D(
        32,
        4,
        strides=2,
        B=B_list[2],
        kernel_initializer=initialization,
    )(z)
    z = ApproxReLU(a, B_list[3])(z)

    z = Conv2D(64, 3, B=B_list[4], padding="same", kernel_initializer=initialization)(z)
    z = ApproxReLU(a, B_list[5])(z)
    z = Conv2D(
        64,
        4,
        strides=2,
        B=B_list[6],
        kernel_initializer=initialization,
    )(z)
    z = ApproxReLU(a, B_list[7])(z)

    z = Flatten()(z)
    z = Dense(
        512,
        B=B_list[9],
        kernel_initializer=initialization,
        kernel_constraint=unit_norm(axis=1),
    )(z)
    z = ApproxReLU(a, B_list[10])(z)
    z = Dense(
        512,
        B=B_list[11],
        kernel_initializer=initialization,
        kernel_constraint=unit_norm(axis=1),
    )(z)
    z = ApproxReLU(a, B_list[12])(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_6C2F(
    input_shape,
    num_classes,
    activation="relu",
    initialization="orthogonal",
):
    x = Input(input_shape)

    z = Conv2D(32, 3, padding="same", kernel_initializer=initialization)(x)
    z = Activation(activation)(z)
    z = Conv2D(32, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        32,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(64, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(64, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        64,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_8C2F(
    input_shape,
    num_classes,
    activation="relu",
    initialization="orthogonal",
):
    x = Input(input_shape)

    z = Conv2D(64, 3, padding="same", kernel_initializer=initialization)(x)
    z = Activation(activation)(z)
    z = Conv2D(64, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        64,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(128, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(128, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        128,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(256, 3, padding="same", kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        256,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def _cnn_CxCC2F(
    backbone_depth,
    backbone_width,
    input_shape,
    num_classes,
    activation="minmax",
    initialization="orthogonal",
    stem_downsample=2,
):
    x = Input(input_shape)

    # Stem.
    z = Conv2D(
        backbone_width,
        5,
        strides=stem_downsample,
        padding="same",
        kernel_initializer=initialization,
    )(x)
    z = Activation(activation)(z)

    # Backbone.
    for _ in range(backbone_depth):
        z = Conv2D(
            backbone_width,
            3,
            padding="same",
            kernel_initializer=initialization,
        )(z)
        z = Activation(activation)(z)

    # Neck.
    z = Conv2D(
        2 * backbone_width,
        4,
        strides=4,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512)(z)
    z = Activation(activation)(z)

    # Head.
    y = Dense(num_classes)(z)

    return x, y


def cnn_C6CC2F(
    input_shape,
    num_classes,
    width=128,
    activation="minmax",
    initialization="orthogonal",
    stem_downsample=2,
):
    return _cnn_CxCC2F(
        6,
        width,
        input_shape,
        num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )


def _liresnet_CxCC2F(
    backbone_depth,
    backbone_width,
    input_shape,
    num_classes,
    activation="minmax",
    initialization="orthogonal",
    stem_downsample=2,
):
    x = Input(input_shape)

    # Stem.
    z = Conv2D(
        backbone_width,
        5,
        strides=stem_downsample,
        padding="same",
        kernel_initializer=initialization,
    )(x)
    z = Activation(activation)(z)

    # Backbone.
    for _ in range(backbone_depth):
        z = LiResNetBlock(
            3,
            residual_scale=backbone_depth ** (-0.5),
            kernel_initializer=initialization,
        )(z)
        z = Activation(activation)(z)

    # Neck.
    z = Conv2D(
        2 * backbone_width,
        4,
        strides=4,
        padding="same",
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512)(z)
    z = Activation(activation)(z)

    # Head.
    y = Dense(num_classes)(z)

    return x, y


def liresnet_C6CC2F(
    input_shape,
    num_classes,
    width=128,
    activation="minmax",
    initialization="orthogonal",
    stem_downsample=2,
):
    return _liresnet_CxCC2F(
        6,
        width,
        input_shape,
        num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )


def liresnet_C18CC2F(
    input_shape,
    num_classes,
    width=256,
    activation="minmax",
    initialization="orthogonal",
    stem_downsample=2,
):
    return _liresnet_CxCC2F(
        18,
        width,
        input_shape,
        num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )
