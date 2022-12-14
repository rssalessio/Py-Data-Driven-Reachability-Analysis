import numpy as np
from pyzonotope import Zonotope, MatrixZonotope, concatenate_zonotope
from pydatadrivenreachability import compute_LTI_matrix_zonotope, LTI_reachability
import scipy.signal as scipysig
from scipy.linalg import null_space, orth

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
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 3)))


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


import pdb
pdb.set_trace()
kernel_basis = null_space(np.vstack((Xm.T, Um.T)))
Bw = (Xp.T - Mw.center) @ kernel_basis
Bw_vec = Bw.flatten()
Aw = []
Aw_vec = []
for i in range(Mw.num_generators):
    Aw.append(Mw.generators[i] @ kernel_basis)
    Aw_vec.append(Aw[-1].flatten())

Aw_vec = np.array(Aw_vec)
beta0 = Bw_vec @ np.linalg.pinv(Aw_vec)
Aw = np.hstack(Aw)
import pdb
pdb.set_trace()
Msigma: MatrixZonotope = compute_LTI_matrix_zonotope(Xm, Xp, Um, Mw)

print(f'Msigma contains [A,B]: {Msigma.contains(np.hstack((A,B)))}')


x_data = LTI_reachability(Msigma, X0, U, W, steps=5, order=5)
x_model = LTI_reachability(scipysig.StateSpace(A,B,C,D), X0, U, W, steps=5, order=5)


for i in range(len(x_data)):
    print('---------------------\n')
    print(f'Step {i}')
    print(x_data[i].center)
    print(x_model[i].center)
