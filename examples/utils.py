import numpy as np
from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope


def concatenate_zonotope(zonotope: Zonotope, N: int) -> MatrixZonotope:
    assert N > 0 and isinstance(N, int), 'N must be a positiveinteger'
    dim_x = zonotope.dim
    C = np.tile(zonotope.center, (dim_x, N))
    G = np.zeros((dim_x, N * N * zonotope.num_generators))

    for i in range(zonotope.num_generators):
        for j in range(N):
            G[:, i*N*N + j*N + j] = zonotope.generators[:, i]
    

    return MatrixZonotope(C, G)


W = Zonotope(np.array(np.zeros((2, 1))), 0.003 * np.ones((2, 1)))


total_samples = 100
M = concatenate_zonotope(W, total_samples)
print(M.generators.shape)