import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from lib.field import *
from lib.nesterov import NesterovMethod


def easy_f(x):
    return x ** 3 + x


def etas(size):
    return sps.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]).rvs(size=size).reshape(size, 2, 1)


def scphere_oracle(x):
    res = np.linalg.norm(x)
    if res < 1:
        return x
    return x / res


def test_on_easy_f(method, steps_num, x_num):
    x_0 = np.array([-0.8, 0])
    my_etas = etas(x_num)
    y_s = np.zeros(x_num)
    for ind, eta in enumerate(my_etas):
        y_s[ind] = np.array(sps.norm.rvs(loc=easy_f(eta.T @ x_0), size=1))
    my_field = SignalField(easy_f, x_num, my_etas, y_s.reshape(x_num, 1), 2)
    results = method(my_field, np.array([0.3, 0.3]), 1 / 10, steps_num, scphere_oracle)
    diffs = np.zeros(steps_num)
    for ind, result in enumerate(results):
        diffs[ind] = np.linalg.norm(result - x_0)
    plt.plot(np.arange(steps_num), diffs)
    plt.show()