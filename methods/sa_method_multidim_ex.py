#!/usr/bin/python3.6

import numpy as np
from scipy.stats import multivariate_normal
from sa_method import sa_method
import matplotlib.pyplot as plt
import random
params = {'legend.fontsize': 15,
          'legend.handlelength': 2,
          'figure.figsize': (15, 10)}
plt.rcParams.update(params)


def run_saa_on_f1(n, m, known_samples_size):
    x_asteriks = np.array([[1]]*n)
    x0 = np.array([[0]]*n)
    modulus_continuity = 1
    steps = 1000
    xs = np.arange(1, steps + 1)
    mult_normal = multivariate_normal(mean=[0]*n, cov=np.identity(n))
    eta_samples = mult_normal.rvs(size=known_samples_size)
    ys_sample = dict()

    def f(x):
        return x**3 + x

    def etas():
        return random.choice(eta_samples).reshape(n, m)

    def ys(eta_arr):
        eta = tuple(map(tuple, eta_arr))
        if eta not in ys_sample:
            ys_sample[eta] = multivariate_normal.rvs(
                mean=f(np.transpose(eta) @ x_asteriks), size=1).reshape(m, 1)
        return ys_sample[eta]

    def G(eta, y, z):
        return eta @ f(np.transpose(eta) @ z) - eta @ y

    def proj(x):
        if np.linalg.norm(x) <= 2:
            return x
        return 2*(x / np.linalg.norm(x))

    x_estim = sa_method(G, modulus_continuity, proj, steps, x0, etas, ys)
    x_diff = np.apply_along_axis(np.linalg.norm, 1, x_estim - x_asteriks)
    plt.plot(xs, x_diff, label=r'$|x^*-x_k|$')
    plt.legend(loc='best')
    plt.xlabel("Итерации")
    plt.ylabel("Разница")
    plt.ylim([0, 0.5])
    plt.title(f'{n}-мерная GLM аппроксимация, стохастический метод.')

    plt.savefig(f'../data/{n}-dim-saa.png')
    # plt.show()


if __name__ == '__main__':
    run_saa_on_f1(1, 1, 100)
    plt.figure(2)
    run_saa_on_f1(2, 1, 100)
    plt.figure(3)
    run_saa_on_f1(3, 1, 100)
