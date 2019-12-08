from field import *


def search_argmin_simple_GD(field, x0=None, mode='precision', L=None, **kwargs):
    if x0 == None:
        x0 = np.zeros(self.dimX)

    if L == None:
        L = (lambda f, z: np.linalg.norm(f(z)) / 2)

    x = x0
    history = [x0]

    if mode == 'precision':
        criteria = lambda : np.linalg.norm(x - history[-1]) < kwargs['precision'] and len(history > 1)

    elif mode == 'steps':
        steps_counted = iter(range(kwargs['steps']))
        criteria = lambda : next(steps_counted, -1) != -1


    while not criteria():
        x -= field(x) / L(f, x)
        history.append(x)

    return x, history




