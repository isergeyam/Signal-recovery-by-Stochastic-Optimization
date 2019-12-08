from field import *

def search_min_simple_GD(field, x0=None, precision=1e-6, steps=10000, mode='precision', L=None):
    if x0 == None:
        x0 = np.zeros(self.dimX)

    if L == None:
        L = (lambda f, z: np.linalg.norm(f(z)) / 2)

    if mode == 'precision':
        Delta = 2*precision
        x = x0
        while Delta >= precision:
            x -= field(x) / L(f, x)

        return




