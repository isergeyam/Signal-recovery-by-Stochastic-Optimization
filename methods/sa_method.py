import numpy as np


def sa_method(G, modulus_continuity, projection_oracle, steps, z0, etas, ys):
    zs = np.array([])
    z = z0
    gamma = 1/modulus_continuity
    for i in range(steps):
        eta = etas()
        z = projection_oracle(z - gamma*G(eta, ys(eta), z))
        zs = np.append(zs, z)
        gamma = 1/((i+2)*modulus_continuity)

    return zs
