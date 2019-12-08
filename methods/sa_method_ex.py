#!/usr/bin/python3.6

import numpy as np
from scipy.stats import norm
from sa_method import sa_method
import matplotlib.pyplot as plt
import random

params = {'legend.fontsize': 15,
          'legend.handlelength': 2,
          'figure.figsize': (15, 10)}
plt.rcParams.update(params)

x_asteriks = 1
modulus_continuity = 1
steps = 1000
xs = np.arange(1, steps + 1)
x0 = 0
eta_samples = norm.rvs(size=100)


def f(x):
    return x**3 + x


def etas():
    return random.choice(eta_samples)


def ys(eta):
    return norm.rvs(loc=f(eta*x_asteriks), size=1)[0]


def G(eta, y, z):
    return eta*f(eta*z) - eta*y


def proj(x):
    if np.linalg.norm(x) <= 2:
        return x
    return 2*(x / np.linalg.norm(x))


plt.plot(xs, np.abs(sa_method(G, modulus_continuity,
                              proj, steps, x0, etas, ys) - x_asteriks), label=r'$|x^*-x_k|$')

plt.legend(loc='best')
plt.xlabel("Iterations")
plt.ylabel("Difference")
plt.ylim([0, 0.5])
plt.title('One dimension GLM approximation.')

plt.savefig('../data/SAA-onedim.png')
plt.show()
