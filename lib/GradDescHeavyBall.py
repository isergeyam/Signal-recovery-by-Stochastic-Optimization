import typing as tp
import math
import numpy as np


def grad_desc_heavy_ball(field, x0, L, steps, projection_oracle, mu = None):
    if not mu:
        mu = L * 5
    alpha = L * 5
    beta = mu
    x0 = projection_oracle(x0)
    x = projection_oracle(x0 - alpha * field(x0))
    history: tp.List[tp.Any] = [x0, x]
    for i in range(steps):
        x = projection_oracle(history[-1] - alpha * field(history[-1]) / (i * 0.5 + 1) + beta * (history[-1] - history[-2]) / (i * 0.5 + 1))
        history.append(x)

    return np.array(history[:-2])
