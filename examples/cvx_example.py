import numpy as np
from pydatadrivenreachability import Zonotope, CVXZonotope, concatenate_zonotope
import scipy.signal as scipysig
import cvxpy as cp
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

dim_x = 7
dim_u = 3

u = cp.Variable((dim_u))

# Initial set and input

X0 = CVXZonotope([1] * dim_x, 0.1 * np.diag([1] * dim_x))
U = CVXZonotope(u,  0.25 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 1)))

X1 = X0.cartesian_product(U)


Mw = concatenate_zonotope(W, 10)
Z = Mw * X1

Zonotope([0] * dim_x, np.zeros((dim_x, 1)))