import numpy as np

class Time_Cts_RNN:
    """Class for time continuous vanilla RNN"""

    def __init__(self, W, activation, g=1):
        """Initialize with parameters, activation function, gain."""

        if W.shape[0] != W.shape[1]:
            raise ValueError('W must be a square matrix')

        self.W = W
        self.activation = activation
        self.g = g

        self.n = self.W.shape[0]
        self.x = np.zeros(self.n)
        self.phi = self.activation.f(self.x)

    def x_dot(self, I=None):
        """Return time derivative in terms of current network
        state and an external input, if provided. Also updates
        the internal value of eta."""

        self.eta = self.W.dot(self.activation.f(self.x))
        ret = -self.x + self.eta
        if I is not None:
            self.I = I
            ret += self.I
        return ret