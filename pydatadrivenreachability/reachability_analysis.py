import numpy as np
from pydatadrivenreachability.zonotope import Zonotope
from pydatadrivenreachability.matrix_zonotope import MatrixZonotope
from typing import Optional, List
from functools import singledispatch
from scipy.signal import StateSpace

def compute_LTI_matrix_zonotope(X0: np.ndarray, X1: np.ndarray, U0: np.ndarray, Mw: MatrixZonotope, Ms: MatrixZonotope) -> MatrixZonotope:
    """
    Computes the matrix zonotope of a linear system given a set of collected trajectories

    See also Theorem 1 in Data-Driven Reachability Analysis Using Matrix Zonotopes, 2021
    Url: https://arxiv.org/pdf/2011.08472.pdf

    :param X0: Collected state-data. Should be a matrix of size (T-1) x dim_x, where T is the number of samples
    :param X1: Collected state-data. Should be a matrix of size (T-1) x dim_x, where T is the number of samples
    :param U0: Collected control-data. Should be a matrix of size (T-1) x dim_u, where T is the number of samples
    :param Mw: Process noise matrix zonotope
    :return: Msigma, the matrix zonotope of the dynamics of the system
    """
    assert X0.shape[0] == X1.shape[0] == U0.shape[0], 'Data has not the same number of samples'
    assert X1.shape[1] == X0.shape[1], 'X data has not the same dimensionality'

    M = Mw + Ms
    Msigma = MatrixZonotope(X1.T - M.center, M.generators)
    return Msigma * np.linalg.pinv(np.vstack((X0.T, U0.T)))

def compute_IO_LTI_matrix_zonotope(
        Y0: np.ndarray,
        Y1: np.ndarray,
        U0: np.ndarray,
        Mw: MatrixZonotope,
        Mv: MatrixZonotope,
        Mav: MatrixZonotope) -> MatrixZonotope:
    """
    Computes the matrix zonotope of a linear system given a set of input/output trajectories
    """
    assert Y0.shape[0] == Y1.shape[0] == U0.shape[0], 'Data has not the same number of samples'
    assert Y1.shape[1] == Y0.shape[1], 'X data has not the same dimensionality'

    M1: MatrixZonotope = -1 * Mw
    M2: MatrixZonotope = -1 * Mv
    Msigma =  (M1 + Y1.T) + M2 + Mav
    return Msigma * np.linalg.pinv(np.vstack((Y0.T, U0.T)))

@singledispatch
def LTI_reachability(sys, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    raise NotImplementedError(f'Unsupported type {type(sys)}')

@singledispatch
def LTI_IO_reachability(sys, Z_y0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, Z_v: Zonotope, Zav: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
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


@LTI_IO_reachability.register(MatrixZonotope)
def _LTI_IO_reachability(sys: MatrixZonotope, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, Z_v: Zonotope, Zav: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    Z_x = [Z_x0]
    for i in range(steps):
        Z_x[i] = Z_x[i].reduce(order)
        Z_x.append(sys * Z_x[i].cartesian_product(Z_u) + Z_w + Z_v + (-1 * Zav))

    Z_x[-1] = Z_x[-1].reduce(order)
    return Z_x

@LTI_IO_reachability.register(StateSpace)
def _LTI_IO_reachability(sys: StateSpace, Z_x0: Zonotope, Z_u: Zonotope, Z_w: Zonotope, Z_v: Zonotope, Zav: Zonotope, steps: int, order: Optional[int] = 10) -> List[Zonotope]:
    Z_x = [Z_x0]
    for i in range(steps):
        Z_x[i] = Z_x[i].reduce(order)
        Z_x.append(Z_x[i] * sys.A + Z_u * sys.B + Z_w + Z_v + (-1 * Zav))

    Z_x[-1] = Z_x[-1].reduce(order)
    return Z_x
