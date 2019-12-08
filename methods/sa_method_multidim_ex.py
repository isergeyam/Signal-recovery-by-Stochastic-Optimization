#!/usr/bin/python3.6

import numpy as np
from scipy.stats import multivariate_normal
from sa_method import sa_method
import matplotlib.pyplot as plt

x_asteriks = np.array([[1], [1]])
modulus_continuity = 1
steps = 1000
xs = np.arange(1, steps + 1)
x0 = np.array([[0], [0]])
mult_normal = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])

n = 2
m = 1


def f(x):
    return x**3 + x


def etas():
    return mult_normal.rvs(size=1).reshape(n, m)


def ys(eta):
    return multivariate_normal.rvs(mean=f(np.transpose(eta) @ x_asteriks),
                                   size=1).reshape(m, 1)


def G(eta, y, z):
    return eta @ f(np.transpose(eta) @ z) - eta @ y


def proj(x):
    if np.linalg.norm(x) <= 2:
        return x
    return 2*(x / np.linalg.norm(x))


x_estim = sa_method(G, modulus_continuity, proj, steps, x0, etas, ys)
x_diff = np.apply_along_axis(np.linalg.norm, 1, x_estim - x_asteriks)
plt.plot(xs, x_diff)

plt.show()
