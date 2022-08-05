import numpy as np
from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope


def concatenate_zonotope(zonotope: Zonotope, N: int) -> MatrixZonotope:
    assert N > 0 and isinstance(N, int), 'N must be a positiveinteger'
    dim_x = zonotope.dim
    C = np.tile(zonotope.center, (dim_x, N))
    G = np.zeros(dim_x, N * N * zonotope.num_generators)

    for i in range(zonotope.num_generators):
        for j in range(N):
            temp_G = np.zeros(dim_x, N)
            G[:, j + i * N] = zonotope.Z[:, i]




    return MatrixZonotope(C, G)


