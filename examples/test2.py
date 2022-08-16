import numpy as np
from pydatadrivenreachability import Zonotope, MatrixZonotope, concatenate_zonotope, compute_LTI_matrix_zonotope, LTI_reachability
import scipy.signal as scipysig
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.ones((5, 1))
C = np.array([1, 0, 0, 0, 0])
D = np.array([0])

dim_x = A.shape[0]
dim_u = 1
dt = 0.05
A,B,C,D,_ = scipysig.cont2discrete(system=(A,B,C,D), dt = dt)

# parameters
trajectories = 1
steps = 120
total_samples = steps * trajectories

# Initial set and input
X0 = Zonotope([1] * dim_x, 0.1 * np.diag([1] * dim_x))
U = Zonotope([10] * dim_u,  0.25 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 1)))


Mw = concatenate_zonotope(W, total_samples - 1)
u = U.sample(total_samples).reshape((trajectories, steps, dim_u))

# Simulate system
X = np.zeros((trajectories, steps, dim_x))
Xm = np.zeros((trajectories, steps - 1, dim_x))
Xp = np.zeros_like(Xm)
for j in range(trajectories):
    X[j, 0, :] = X0.sample()
    for i in range(1, steps):
        X[j, i, :] = A @ X[j, i - 1, :] +  np.squeeze(B * u[j, i - 1]) + W.sample()


Xm = np.reshape(X[:,:-1,:], ((steps - 1) * trajectories, dim_x))
Xp = np.reshape(X[:, 1:,:], ((steps - 1) * trajectories, dim_x))
Um = np.reshape(u[:, :-1,:], ((steps - 1) * trajectories, dim_u))

Msigma = compute_LTI_matrix_zonotope(Xm, Xp, Um, Mw)


import cvxpy as cp
