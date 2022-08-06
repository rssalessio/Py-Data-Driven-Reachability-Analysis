import numpy as np
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
dt = 0.05
A,B,C,D,_ = scipysig.cont2discrete(system=(A,B,C,D), dt = dt)


# parameters
initpoints = 1
steps = 120
total_samples = steps * initpoints

# Initial set and input
X0 = Zonotope(np.ones((dim_x, 1)), 0.1 * np.diag([1] * dim_x))
U = Zonotope(np.ones((1, 1)),  0.25 * np.ones((1,1)))
W = Zonotope(np.zeros((dim_x, 1)), 0.003 * np.ones((dim_x, 1)))

Mw = concatenate_zonotope(W, total_samples)
u = U.sample(total_samples)

# Simulate system
X = np.zeros((total_samples, dim_x))
for j in range(initpoints):
    X[j * steps, :] = X0.sample()
    for i in range(1, steps):
        print(f'{j}-{i}')
        X[j * steps + i, :] = A @ X[j * steps + i -1, :] +  B @ u[j * steps + i - 1] + W.sample()

print(X)