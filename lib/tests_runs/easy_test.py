import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

import field

def easy_f(x):
    return x**3 + x

def log_f(x):
    return 1 / (1 + np.exp(-x))

def etas(size, dim):
    return sps.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim)).rvs(size=size).reshape((size, dim, 1))

def sphere_oracle(x):
    res = np.linalg.norm(x)
    if res < 2:
        return x
    return x / res * 2

def test_on_f(func, method, steps_num, x_num, graph_name, figname):
    x_0 = np.array([-0.8, 0])
    my_etas = etas(x_num, 2)
    y_s = np.zeros(x_num)
    for ind, eta in enumerate(my_etas):
        y_s[ind] = np.array(sps.norm.rvs(loc=easy_f(eta.T @ x_0), size=1))
    my_field = field.SignalField(
        func, x_num, my_etas, y_s.reshape(x_num, 1), 2)
    results = method(my_field, np.array(
        [0.3, 0.3]), 1/10, steps_num, sphere_oracle)
    diffs = np.zeros(steps_num)
    for ind, result in enumerate(results):
        diffs[ind] = np.linalg.norm(result - x_0)
    plt.plot(np.arange(steps_num), diffs)
    plt.xlabel("Количество шагов")
    plt.ylabel("Отклонение от истинного x")
    plt.title(graph_name)
    plt.savefig(figname)
    plt.show()

def test_on_logregr(method, steps_num, x_num, graph_name, figname):
    x_0 = np.array([-0.8, 0])
    my_etas = etas(x_num, 2)
    y_s = np.zeros(x_num)
    for ind, eta in enumerate(my_etas):
        y_s[ind] = np.array(sps.bernoulli.rvs(p=log_f(eta.T @ x_0), size=1))
    my_field = field.SignalField(log_f, x_num, my_etas, y_s.reshape(x_num, 1), 2)
    results = method(my_field, np.array(
        [0.3, 0.3]), 1/10, steps_num, sphere_oracle)
    diffs = np.zeros(steps_num)
    for ind, result in enumerate(results):
        diffs[ind] = np.linalg.norm(result - x_0)
    plt.plot(np.arange(steps_num), diffs)
    plt.xlabel("Количество шагов")
    plt.ylabel("Отклонение от истинного x")
    plt.title(graph_name)
    plt.savefig(figname)
    plt.show()

def test_on_logreg_dim(method, steps_num, x_num, graph_name, figname, dim):
    if dim < 4:
        dim = 4
    x_0 = np.zeros(dim)
    x_0[0] = 0.6
    x_0[3] = 0.2
    x_0[2] = 0.1
    my_etas = etas(x_num, dim)
    y_s = np.zeros(x_num)
    for ind, eta in enumerate(my_etas):
        y_s[ind] = np.array(sps.bernoulli.rvs(p=log_f(eta.T @ x_0), size=1))
    my_field = field.SignalField(log_f, x_num, my_etas, y_s.reshape(x_num, 1), dim)
    results = method(my_field, np.zeros(dim), 1, steps_num, sphere_oracle)
    diffs = np.zeros(steps_num)
    for ind, result in enumerate(results):
        diffs[ind] = np.linalg.norm(result - x_0)
    plt.plot(np.arange(steps_num), diffs)
    plt.xlabel("Количество шагов")
    plt.ylabel("Отклонение от истинного x")
    plt.title(graph_name)
    plt.savefig(figname)
    plt.show()

def test_on_easy_f(method, steps_num, x_num, graph_name, figname):
    test_on_f(easy_f, method, steps_num, x_num, graph_name, figname)