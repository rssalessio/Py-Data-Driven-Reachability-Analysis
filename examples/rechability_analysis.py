import numpy as np
from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope


def compute_linear_system_matrix_zonotope(X0: np.ndarray, X1: np.ndarray, U0: np.ndarray, Mw: MatrixZonotope) -> MatrixZonotope:
    """
    Computes the matrix zonotope of a linear system given a set of collected trajectories

    See also Theorem 1 in Data-Driven Reachability Analysis Using Matrix Zonotopes, 2021
    Url: https://arxiv.org/pdf/2011.08472.pdf
    """
    assert X0.shape[0] == X1.shape[0] == U0.shape[0], 'Data has not the same number of samples'
    assert X1.shape[1] == X0.shape[1], 'X data has not the same dimensionality'

    Mx = MatrixZonotope(X1.T - Mw.center, Mw.generators)
    return Mx * np.linalg.pinv(np.vstack((X0.T, U0.T)))
