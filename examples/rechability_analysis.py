import numpy as np
from zonotope import Zonotope
from matrix_zonotope import MatrixZonotope
from typing import Optional, List
from functools import singledispatch
from scipy.signal import StateSpace

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

@singledispatch
def LTI_reachability(sys, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    raise NotImplementedError(f'Unsupported type {type(sys)}')

@LTI_reachability.register(MatrixZonotope)
def _LTI_reachability(sys: MatrixZonotope, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    Z_x = [Z_x0]
    for i in range(steps):
        Z_x[i] = Z_x[i].reduce(order)
        Z_x.append(sys * Z_x[i].cartesian_product(Z_u) + Z_w)

    Z_x[-1] = Z_x[-1].reduce(order)
    return Z_x


@LTI_reachability.register(StateSpace)
def _LTI_reachability(sys: StateSpace, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    Z_x = [Z_x0]
    for i in range(steps):
        Z_x[i] = Z_x[i].reduce(order)
        Z_x.append(Z_x[i] * sys.A + Z_u * sys.B + Z_w)

    Z_x[-1] = Z_x[-1].reduce(order)
    return Z_x
