import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

from lib.field import *
from lib.nesterov import NesterovMethod

def easy_f(x):
  return x^3 + x

def etas(size):
  return sps.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]]).rvs(size = size).reshape(3,2,1)

def test_on_easy_f(method, steps_num, x_num):
  x_0 = np.array([0.5, 0.5])
  projection_oracle = lambda x: x / np.abs(x)
  my_etas = etas(x_num)
  y_s = np.zeros(x_num)
  for ind, eta in enumerate(etas):
    y_s[ind] = sps.norm.rvs(loc = easy_f(eta & x_0), size=1)
  my_field = SignalField(easy_f, x_num, etas, y_s, 2)
  results = method(my_field, np.array([]))