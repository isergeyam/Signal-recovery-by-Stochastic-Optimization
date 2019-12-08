import numpy as np


def _search_argmin_simple_GD(f, x0, mode='precision', L=None, **kwargs):
    if L == None:
        L = np.linalg.norm(f(x0))

    proj = kwargs['proj_oracle']
    x = proj(x0)

    history = [x.copy()]

    criteria = lambda : True

    if mode == 'precision':
        criteria = lambda : (np.linalg.norm(x - history[-1]) < kwargs['precision'] and len(history) > 1)

    elif mode == 'steps':
        steps_counted = iter(range(kwargs['steps']))
        criteria = lambda : next(steps_counted, -1) != -1


    while not criteria():
        history.append(x.copy())
        L = max(L, np.linalg.norm(f(x)))
        x -= f(x) / (2 * L)
        x = proj(x)


    return (x, np.array(history)[1:])


def SimpleGDForMonotoneFields(field, start, steps_scale, steps_num, proj_oracle):
    return _search_argmin_simple_GD(field, start, mode='steps', L=steps_scale, **{'steps':steps_num, 'proj_oracle':proj_oracle})[1]

#if __name__ == '__main__':
#    print(search_argmin_simple_GD((lambda x: x), np.ones(2), mode='precision', **{'precision':1e-4}))
#    print(search_argmin_simple_GD((lambda x: x), np.ones(2), mode='precision', **{'precision':1e-8}))


