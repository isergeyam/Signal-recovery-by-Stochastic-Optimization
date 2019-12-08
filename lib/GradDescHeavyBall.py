import typing as tp
import math
import numpy as np


def grad_desc_heavy_ball(field, x0, L, steps, projection_oracle, mu = None):
    if not mu:
        mu = L * 5
    # L = 1 / L / 2
    # mu = 1 / mu / 2
    # print(L, mu)
    # k = L / mu
    # alpha = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    alpha = L
    # beta = ((math.sqrt(k) - 1) / (math.sqrt(k) + 1)) ** 2
    beta = mu
    x0 = projection_oracle(x0)
    x = projection_oracle(x0 - alpha * field(x0))
    history: tp.List[tp.Any] = [x0, x]
    for _ in range(steps):
        x = projection_oracle(history[-1] - alpha * field(history[-1]) + beta * (history[-1] - history[-2]))
        history.append(x)

    return np.array(history[:-2])
