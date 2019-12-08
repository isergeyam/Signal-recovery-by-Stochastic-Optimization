import numpy as np
from lib.tests_runs.easy_test import test_on_easy_f

def NesterovMethod(field, start, steps_scale, steps_num, projection_oracle):
    result = []
    x = start
    y = start
    for k in range(steps_num):
        new_x = projection_oracle(y - steps_scale / 5 * field(y))
        y = projection_oracle(new_x + k / (k + 3) * (new_x - x))
        x = new_x
        result.append(x)
    return np.array(result)

if __name__ == "__main__":
    test_on_easy_f(NesterovMethod, 1000, 100, "Метод Нестрова", "nesterov.png")