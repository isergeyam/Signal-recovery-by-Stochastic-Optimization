from tests_runs.easy_test import test_on_easy_f
from field import *
from GradDescHeavyBall import grad_desc_heavy_ball

test_on_easy_f(grad_desc_heavy_ball, 1000, 10, 'BALLS')
