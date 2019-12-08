from field import Field
import typing as tp
import math
import numpy as np


def quasi_newton_method(field, x0: tp.Any, alpha, steps, projection_oracle, mode='BFGS') -> tp.Any:

    def SR1(s, y, H):
        Hy = np.matmul(H, y)
        return H + (s - Hy) * np.transpose(s - Hy) / (np.dot(s - Hy, y))

    def DFP(s, y, H):
        Hy = H @ y
        return H - (Hy @ y.T @ H) / np.dot(Hy, y) + (s @ s.T) / np.dot(y, s)

    def BFGS(s, y, H):
        E = np.identity(s.size)
        Hy = H @ y
        return (E - (s @ y.T) / np.dot(y, s)) @ H @ (E - (y @ s.T) / np.dot(y, s)) + (s @ s.T) / np.dot(y, s)

    if mode == 'SR1':
        update = SR1
    elif mode == 'DFP':
        update = DFP
    else:
        update = BFGS

    x0 = projection_oracle(x0)
    x = x0
    H = np.identity(x0.size)
    history: tp.List[tp.Any] = [x0]
    for i in range(steps):
        old_x = x.copy()
        x = x - alpha * H @ field(x) / 10
        H = update(x - old_x, field(x) - field(old_x), H)
        x = projection_oracle(x)
        history.append(x)

    return history[:-1]

