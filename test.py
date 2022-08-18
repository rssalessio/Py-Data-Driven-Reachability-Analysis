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

Msigma: MatrixZonotope = compute_LTI_matrix_zonotope(Xm, Xp, Um, Mw)

print(f'Msigma contains [A,B]: {Msigma.contains(np.hstack((A,B)))}')

import cvxpy as cp
import dccp


A0 = Msigma.center[:, :dim_x]
B0 = Msigma.center[:, dim_x:]
K = cp.Variable((dim_u, dim_x))
A = cp.Variable((dim_x, dim_x))
F = cp.Variable((dim_x, dim_x))

beta_A = cp.Variable(Msigma.num_generators)


objective = cp.norm(A0+B0@K) + cp.norm(A-A0 +F- B0 @ K)
constraints = [
    beta_A >= -1, beta_A <= 1 
]

Agen = Msigma.center[:, :dim_x]
for i in range(Msigma.num_generators):
    Agen = Agen + Msigma.generators[i][:, :dim_x] * beta_A[i]

constraints.append(A == Agen)


problem = cp.Problem(cp.Minimize(objective), constraints)
res = problem.solve(solver=cp.MOSEK, verbose=True)
print(f'Result: {res}')
print(K.value)
print(A.value)
print(F.value)
F=F.value
B=F@np.linalg.pinv(K.value)
print(f'Msigma contains [A,B]: {Msigma.contains(np.hstack((A.value,B)))}')

# myprob = cvx.Problem(cvx.Maximize(cvx.norm(x - y,2)), [0 <= x, x <= 1, 0 <= y, y <= 1])
# print("problem is DCP:", myprob.is_dcp())   # false
# print("problem is DCCP:", dccp.is_dccp(myprob))  # true
# result = myprob.solve(method='dccp')
# print("x =", x.value)
# print("y =", y.value)
# print("cost value =", result[0])

# print(Msigma.sample(100).shape)
X = Msigma.sample(1000)
Y = Msigma.sample(1000)
norms = []
for i in range(X.shape[0]):
    norms.append(np.linalg.norm(X[i] - Y[i]))

import matplotlib.pyplot as plt
plt.plot(norms)

plt.show()
solutions = []


for i in [1, 5, 10, 20, 50, 100]:
    print(f'Steps {i}')
    A1 = cp.Variable((dim_x, dim_x))
    A2 = cp.Variable((dim_x, dim_x))

    beta_A0 = cp.Variable(Msigma.num_generators)
    beta_A1 = cp.Variable(Msigma.num_generators)

    objective = cp.norm(A1-A2)
    constraints = [
        beta_A0 >= -1, beta_A0 <= 1,
        beta_A1 >= -1, beta_A1 <= 1 
    ]

    Agen0 = Msigma.center[:, :dim_x]
    Agen1 = Msigma.center[:, :dim_x]
    for i in range(Msigma.num_generators):
        Agen0 = Agen0 + Msigma.generators[i][:, :dim_x] * beta_A0[i]
        Agen1 = Agen1 + Msigma.generators[i][:, :dim_x] * beta_A1[i]

    constraints.append(A1 == Agen0)
    constraints.append(A2 == Agen1)
    problem = cp.Problem(cp.Maximize(objective), constraints)

    res = problem.solve(method='dccp', solver=cp.ECOS, verbose=False, ccp_times=i, warm_start=False)
    solutions.append(res[0])
    print(f'Result: {res[0]}')
    print('---------')
    # print("problem is DCCP:", dccp.is_dccp(problem)) 
    # print(A1.value)
    # print(A2.value)

plt.plot(solutions)
plt.show()
