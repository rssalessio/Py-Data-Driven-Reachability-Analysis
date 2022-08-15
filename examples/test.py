import numpy as np
import cvxpy as cp

A = np.array([
    [1, 0, 1,2],
    [0, 3, 1,0]
])

b = np.array([2, 3])

beta=cp.Variable(A.shape[1])
constraints = [beta >= -1, beta <= 1]
problem = cp.Problem(cp.Minimize(cp.norm(A@beta -b)), constraints)
res = problem.solve()
print(res)
print(beta.value)

Img = np.linalg.svd(A)[0][:2]
Ker = np.linalg.svd(A)[2][2:]
print(Img.shape)
print(Ker.shape)


beta_null=  np.linalg.pinv(Ker) @ np.array([1, 1])
beta_img = np.linalg.pinv(A) @ b
beta_null_2 = beta.value - beta_img
print(A @ beta_null_2)
print(Ker @ beta_img)
print(A @ beta_null)
print(A @ (beta_img + beta_null))
theta = Ker @ (np.random.uniform(low=-1, high=1, size=(4,1)).flatten()  - beta_img)
print(theta)
gen = beta_img + (np.linalg.pinv(Ker) @ theta).flatten()
import pdb
pdb.set_trace()