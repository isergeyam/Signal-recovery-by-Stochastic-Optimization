import numpy as np

class Field:
    def __init__(self, f, dimX, dimY):
        self.f = f
        self.dimX = dimX
        self.dimY = dimY

    def _search_min_simple_GD(self, x0=None, precision=1e-6):
        if x0 == None:
            x0 = np.zeros(self.dimX)
