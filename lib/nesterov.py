import numpy as np

def NesterovMethod(field, start, steps_scale, steps_num, projection_oracle):
  result = np.ones(steps_scale)
  x = start
  y = start
  for k in range(steps_num):
    new_x = y - steps_scale* field(y)
    y = new_x + (k) / (k + 3) * (new_x - x)
    x = new_x
    result[k] = x
  return result