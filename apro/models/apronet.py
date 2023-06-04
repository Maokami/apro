import tensorflow as tf
import numpy as np
from gloro.models import GloroNet

from tensorflow.keras.models import Model


class AproNet(GloroNet):
    def __init__(
        self,
        inputs=None,
        outputs=None,
        epsilon=None,
        *,
        _lc_frozen=False,
        _hardcoded_lc=1,
        _skip_init=False,
        **kwargs,
    ):
        # Validate the provided parameters.
        if epsilon is None:
            raise ValueError("`epsilon` is required")

        if inputs is None or outputs is None:
            raise ValueError("must specify `inputs` and `outputs`")

        model = Model(inputs, outputs)

        super().__init__(
            None,
            None,
            epsilon,
            model=model,
            _lc_frozen=_lc_frozen,
            _hardcoded_lc=_hardcoded_lc,
            _skip_init=_skip_init,
            **kwargs,
        )
        self._lipschitz_computers_inf = [lambda: 1.0] + [
            layer.lipschitz_inf for layer in model.layers[1:-1]
        ]

    @property
    def sub_lipschitz(self):
        """
        This is an upper bound on the Lipschitz(inf) constant up to the penultimate
        layer.
        """
        return tf.reduce_prod(
            [lipschitz() for lipschitz in self._lipschitz_computers_inf]
        )

    def _K_ij(self, k, j):
        """
        K_ij is the Lipschitz constant on the margin y_j - y_i.
        """
        kW = k * self.layers[-1].kernel

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:, :, None] - kW[None]

        return tf.norm(kW_ij, ord=np.inf, keepdims=False, axis=1)

    def compute_total_error(self, initial_error):
        error = initial_error
        for layer in self.f.layers[1:-1]:
            error = layer.propagate_error(error)
        return error

    def compute_bound(self):
        B = 1
        B_list = []
        B_list.append(B)
        for layer in self.f.layers[1:-1]:
            B = layer.bound(B)
            B_list.append(B.numpy())
        return B_list

    def set_B(self, B_list):
        for layer, B in zip(self.f.layers[1:-1], B_list):
            layer.set_B(B)
