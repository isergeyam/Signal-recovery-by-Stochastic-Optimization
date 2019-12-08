import numpy as np

def NesterovMethod(field, start, steps_scale, steps_num, projection_oracle):
  result = []
  x = start
  y = start
  for k in range(steps_num):
    new_x = projection_oracle(y - steps_scale * field(y))
    y = projection_oracle(new_x + k / (k + 3) * (new_x - x))
    x = new_x
    result.append(x)
  return np.array(result)