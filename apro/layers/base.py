from abc import abstractmethod

from gloro.layers import GloroLayer


class AproLayer(GloroLayer):
    @abstractmethod
    def lipschitz_inf(self):
        """
        Returns the Lipschitz constant (inf norm) of this layer.
        """
        raise NotImplementedError

    @abstractmethod
    def propagate_error(self, error):
        """
        Returns the error propagation from previous error.
        """
        raise NotImplementedError

    @abstractmethod
    def bound(self, input_bound):
        """
        Returns the output bound from input bound.
        """
        raise NotImplementedError
