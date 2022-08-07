import numpy as np
import numpy as np
from pydatadrivenreachability import Zonotope, MatrixZonotope, concatenate_zonotope, compute_IO_LTI_matrix_zonotope, LTI_IO_reachability
import scipy.signal as scipysig
import scipy.signal as scipysig
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.ones((5, 1))
C = np.eye(A.shape[0])
D = np.zeros((5, 1))

dim_x = A.shape[0]
dim_y = len(C)
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
V = Zonotope([0] * dim_x, 0.002 * np.ones((dim_x, 1)))
AV = V * A


Mw = concatenate_zonotope(W, total_samples - 1)
Mv = concatenate_zonotope(V, total_samples - 1)
Mav = Mv * A # or Mav = concatenate_zonotope(AV, total_samples - 1)


u = U.sample(total_samples).reshape((trajectories, steps, dim_u))

# Simulate system
X = np.zeros((trajectories, steps, dim_x))
Y = np.zeros((trajectories, steps, dim_y))
Ym = np.zeros((trajectories, steps - 1, dim_y))
Yp = np.zeros_like(Ym)
for j in range(trajectories):
    X[j, 0, :] = X0.sample()
    for i in range(1, steps):
        X[j, i, :] = A @ X[j, i - 1, :] +  np.squeeze(B * u[j, i - 1]) + W.sample()
        Y[j, i, :] = C @ X[j, i, :] + V.sample()


Ym = np.reshape(X[:,:-1,:], ((steps - 1) * trajectories, dim_y))
Yp = np.reshape(X[:, 1:,:], ((steps - 1) * trajectories, dim_y))
Um = np.reshape(u[:, :-1,:], ((steps - 1) * trajectories, dim_u))

Msigma = compute_IO_LTI_matrix_zonotope(Ym, Yp, Um, Mw, Mv, Mav)

print(f'Msigma contains [A,B]: {Msigma.contains(np.hstack((A,B)))}')



y_data = LTI_IO_reachability(Msigma, X0, U, W, V, AV, steps=5, order=5)
y_model = LTI_IO_reachability(scipysig.StateSpace(A,B,C,D), X0, U, W, V, AV, steps=5, order=5)

print(y_data[-1].shape)
print(y_model[-1].shape)

for i in range(len(y_data)):
    print('---------------------\n')
    print(f'Step {i}')
    print(y_data[i].center)
    print(y_model[i].center)

