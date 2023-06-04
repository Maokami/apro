import tensorflow as tf
import numpy as np

import architectures
import utils

from apro.models import AproNet

train, test, metadata = utils.get_data("mnist", 512, "standard")

input_shape = metadata.features["image"].shape
num_classes = metadata.features["label"].num_classes
weigh_path = "../pretrained_model/mnist/mnist_cnn_4c3f_e0.h5"
x, y = architectures.cnn_4C3F(input_shape, num_classes, 7)
g = AproNet(x, y, epsilon=1 / 255)

g.f.load_weights(weigh_path)

for layer in g.layers[1:]:
    print(layer.name)
    print(layer.lipschitz_inf())

i = 12
for input, label in train:
    target_layer = g.layers[i]

    for layer in g.layers[1:i]:
        input = layer(input)
    output = target_layer(input)

    norm_i = tf.norm(input, ord=np.inf)
    norm_o = tf.norm(output, ord=np.inf)

    lipschitz = target_layer.lipschitz_inf()
    actual = norm_o / norm_i
    print(f"target layer: {target_layer.name}")
    print("lipschitz: ", lipschitz.numpy())
    print("actual:", actual.numpy())
