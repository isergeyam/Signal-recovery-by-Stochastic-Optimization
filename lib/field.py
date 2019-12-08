import numpy as np
import scipy as sp

class Field:
    def __init__(self, f, dimX, dimY):
        self.f = f
        self.dimX = dimX
        self.dimY = dimY


class SignalField(object):
    def __init__(self, f, K, etas, ys, dim):
        self.f = f
        self.etas = etas
        self.dim = dim
        self.K = K
        sum = np.zeros(dim)
        for ind, eta in enumerate(etas):
            sum = sum + eta @ ys[ind]
        sum = sum / K
        self.presum = sum

    def __call__(self, vector):
        res = np.zeros(self.dim)
        for eta in self.etas:
            res = res + eta @ self.f(eta.T @ vector)
        return res / self.K - self.presum

