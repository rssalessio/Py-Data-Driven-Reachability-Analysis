import numpy as np
from pydatadrivenreachability import Zonotope, MatrixZonotope, concatenate_zonotope, compute_LTI_matrix_zonotope, LTI_reachability
import scipy.signal as scipysig
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


dim_x = 2
# Initial set and input
generators = np.array([
    [1, 0, 2, 0],
    [0, 1, 1, 2]
])
X0 = Zonotope([0] * dim_x, generators)

print(X0.max_num_vertices)
print(X0.compute_vertices())

print(X0.over_approximate().interval.left_limit)
print(X0.over_approximate().interval.right_limit)
print(X0.over_approximate().compute_vertices())