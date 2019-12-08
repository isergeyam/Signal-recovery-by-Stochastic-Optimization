from lib.field import Field
import typing as tp
import math


def grad_desc_heavy_ball(field: Field, x0: tp.Any, L, steps, projection_oracle, mu = 1) -> tp.Any:
    k = L / mu
    alpha = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    beta  = ((math.sqrt(k) - 1) / (math.sqrt(k) + 1)) ** 2
    x = x0 - alpha * field(x0)
    history: tp.List[tp.Any] = [x0, x]
    for i in range(steps):
        x = history[-1] + alpha * field(history[-1]) + beta * (history[-1] - history[-2])
        history.append(x)

    return x, history