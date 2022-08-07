import numpy as np
from rechability_analysis import compute_linear_system_matrix_zonotope, LTI_reachability
from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope
from utils import concatenate_zonotope
import scipy.signal as scipysig

A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.zeros((5, 1))
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
X0 = Zonotope(np.ones((dim_x, 1)), 0.1 * np.diag([1] * dim_x))
U = Zonotope(10*np.ones((dim_u, 1)),  0.25 * np.diag([1] * dim_u))
W = Zonotope(np.zeros((dim_x, 1)), 0.005 * np.ones((dim_x, 1)))

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

Msigma = compute_linear_system_matrix_zonotope(Xm, Xp, Um, Mw)

print(f'Msigma contains [A,B]: {Msigma.contains(np.hstack((A,B)))}')

test = Zonotope(np.zeros((2, 1)), np.diag([1] * 2))
print(test.contains(np.array([1,1])))


x_model = LTI_reachability(scipysig.StateSpace(A,B,C,D), X0, U, W, steps=6, order=5)
x_data = LTI_reachability(Msigma, X0, U, W, steps=6, order=5)

print(x_data[-1].shape)
print(x_model[-1].shape)

for i in range(len(x_data)):
    print('---------------------\n')
    print(f'Step {i}')
    print(x_data[i].center)
    print(x_model[i].center)