#!/usr/bin/python3.6

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from sa_method import sa_method
import matplotlib.pyplot as plt
import random
params = {'legend.fontsize': 15,
          'legend.handlelength': 2,
          'figure.figsize': (15, 10)}
plt.rcParams.update(params)


def run_saa_on_func(n, m, known_samples_size, steps, f, f_tex, f_name,
                    y_theta, modulus_continuity):
    x_asteriks = np.array([[1]]*n)
    x0 = np.array([[0]]*n)
    xs = np.arange(1, steps + 1)
    mult_normal = multivariate_normal(mean=[0]*n, cov=np.identity(n))
    eta_samples = mult_normal.rvs(size=known_samples_size)
    ys_sample = dict()

    def etas():
        return random.choice(eta_samples).reshape(n, m)

    def ys(eta_arr):
        eta = tuple(map(tuple, eta_arr))
        if eta not in ys_sample:
            ys_sample[eta] = y_theta(f(np.transpose(eta) @ x_asteriks))
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
    plt.title(f'{n}-мерная GLM аппроксимация, стохастический метод для {f_tex}')

    # plt.savefig(f'../data/{n}-dim-saa-{f_name}.png')
    plt.show()


if __name__ == '__main__':
    def f1(x): return x**3 + x
    def y1_theta(x): return multivariate_normal.rvs(
        mean=x, size=1).reshape(1, 1)

    f1_tex = r'$x + x^3$'
    run_saa_on_func(1, 1, 100, 1000, f1, f1_tex, 'f1', y1_theta, 1)
    plt.figure(2)
    run_saa_on_func(2, 1, 100, 1000, f1, f1_tex, 'f1', y1_theta, 1)
    plt.figure(3)
    run_saa_on_func(3, 1, 100, 1000, f1, f1_tex, 'f1', y1_theta, 1)

    def f2(x):
        res = 1/(1 + np.exp(-x))
        return res
    f2_tex = r'$(1 + e^{-x})^{-1}$'

    def y2_theta(x):
        return bernoulli.rvs(x[0][0], size=1).reshape(1, 1)

    plt.figure(4)
    run_saa_on_func(1, 1, 100000, 100000, f2, f2_tex,
                    'regression', y2_theta, 0.1)
    plt.figure(5)
    run_saa_on_func(2, 1, 100000, 100000, f2, f2_tex,
                    'regression', y2_theta, 0.1)
    plt.figure(6)
    run_saa_on_func(3, 1, 100000, 100000, f2, f2_tex,
                    'regression', y2_theta, 0.1)
