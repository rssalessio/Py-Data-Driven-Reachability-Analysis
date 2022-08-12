from __future__ import annotations
from email import generator
from typing import List, Union, Tuple
import numpy as np
from pydatadrivenreachability.zonotope import Zonotope
from copy import deepcopy
from pydatadrivenreachability.interval_matrix import IntervalMatrix
from pydatadrivenreachability.cvx_zonotope import CVXZonotope
import cvxpy as cp

class MatrixZonotope(object):
    """
    MatZonotope class.
    
    Represents the set of matrices of dimension (n,p) contained in the zonotope
    
    See also CORA Library
    https://github.com/TUMcps/CORA/blob/master/matrixSet/%40matZonotope/matZonotope.m
    """
    dim_n: int
    dim_p: int
    num_generators: int
    
    """ Matrix of dimension (n,p) """
    _center: np.ndarray

    """ Tensor of dimension (g,n,p), where g is the number of generators"""
    _generators: np.ndarray

    def __init__(self, center: np.ndarray, generators: np.ndarray):
        assert len(center.shape) == 2 and len(generators.shape) == 3, \
            "Center must be a matrix and generators a tensor"
        assert center.shape == generators.shape[1:], \
            "Center and generators must have the same dimensions"
        self._center = center
        self._generators = generators

        self.dim_n = self._center.shape[0]
        self.dim_p = self._center.shape[1]
        self.num_generators = self._generators.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.dim_n, self.dim_p)

    @property
    def center(self) -> np.ndarray:
        """ Returns the center of the matrix zonotope, of dimension (n,p)"""
        return self._center
    
    @property
    def generators(self) -> np.ndarray:
        """ Returns the generators of the matrix zonotope, of dimension (g,n,p) where g is the number of generators """
        return self._generators
    
    @property
    def order(self) -> float:
        """ Returns the order of the matrix zonotope.
            Equivalent to converting the matrix zonotope to a zonotope and then computing the order of this zonotope
        """
        return self.num_generators / np.prod(self.shape)

    def copy(self) -> MatrixZonotope:
        return MatrixZonotope(deepcopy(self.center), deepcopy(self.generators))

    def __add__(self, operand: Union[float, int, np.ndarray, MatrixZonotope]) -> MatrixZonotope:
        if(isinstance(operand, np.ndarray)) or isinstance(operand, int) or isinstance(operand, float):
            if isinstance(operand, np.ndarray):
                assert operand.shape == self.center.shape, 'Incorrect shape for operand'
            return MatrixZonotope(self.center + operand, self.generators)
        elif isinstance(operand, MatrixZonotope):
            assert self.shape == operand.shape, 'Incorrect dimensionality for operand'
            return MatrixZonotope(self.center + operand.center, np.concatenate((self.generators, operand.generators), axis=0))
        else:
            raise NotImplementedError(f'Add not implemented for {type(operand)}')

    __radd__ = __add__

    def __mul__(self, operand: Union[int, float, np.ndarray, Zonotope, CVXZonotope]) -> Union[MatrixZonotope, CVXZonotope, Zonotope]:
        if isinstance(operand, float) or isinstance(operand, int):
            return MatrixZonotope(self.center * operand, self.generators * operand)
        
        elif isinstance(operand, Zonotope):
            assert self.center.shape[1] == operand.Z.shape[0]
            Znew = self.center @ operand.Z

            for i in range(self.num_generators):
                Zadd = self.generators[i] @ operand.Z
                Znew = np.hstack((Znew, Zadd))
            return Zonotope(Znew[:,:1], Znew[:, 1:])

        elif isinstance(operand, CVXZonotope):
            assert self.center.shape[1] == operand.Z.shape[0]
            Znew: List[cp.Expression] = [self.center @ operand.Z]

            for i in range(self.num_generators):
                Zadd: cp.Expression = self.generators[i] @ operand.Z
                Znew.append(Zadd)

            Znew = cp.hstack(Znew)
            return CVXZonotope(Znew[:,:1], Znew[:, 1:])

        elif isinstance(operand, np.ndarray):
            # Left multiplication, operand * self
            if operand.shape[1] == self.center.shape[0]:
                center = operand @ self.center
                generators = np.zeros((self.num_generators, operand.shape[0], self.dim_p))
                
                for i in range(self.num_generators):
                    generators[i, :, :] = operand @ self.generators[i]

            # Right multiplication, self * operand
            elif operand.shape[0] == self.center.shape[1]:
                center = self.center @ operand
                generators = np.zeros((self.num_generators, self.dim_n, operand.shape[1]))
                
                for i in range(self.num_generators):
                    generators[i, :, :] = self.generators[i] @ operand
            
            else:
                raise Exception('Invalid dimension')
            
            return MatrixZonotope(center, generators)
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    def __str__(self):
        return f'Center: {self.center} - Generators: {self.generators.T}'

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates a uniform random points within a matrix zonotope

        Return a tensor of size (b x n x p), where (n,p) is the dimensionality
        of a single sample, and b is the number of samples

        :param batch_size: number of random points
        :return: A tensor where each element is a sample
        """
        beta = np.random.uniform(low=-1, high=1, size=(batch_size, self.num_generators))
        return (self.center[None, :] + np.tensordot(beta, self.generators, axes=1))

    @property
    def interval_matrix(self) -> IntervalMatrix:

        delta = np.abs(self.generators[0])
        for i in range(1, self.num_generators):
            delta += np.abs(self.generators[i])

        return IntervalMatrix(self.center, delta)

    def contains(self, X: np.ndarray) -> bool:
        """
        Returns true if the matrix zonotope contains X
        """
        return self.interval_matrix.contains(X)

    def over_approximate(self) -> MatrixZonotope:
        """
        Over approximates a Matrix zonotope by a cube
        """
        delta = np.abs(self.generators[0])
        for i in range(1, self.num_generators):
            delta += np.abs(self.generators[i])
        return MatrixZonotope(self.center, delta[None,:])

    @property
    def zonotope(self) -> Zonotope:
        """ Convert the matrix zonotope to a zonotope """
        center = self.center.flatten()
        generators = [self.generators[i].flatten()[:, None] for i in range(self.num_generators)]
        return Zonotope(center, np.hstack(generators))
    
    def reduce(self, order: int) -> MatrixZonotope:
        """
        Reduces the matrix zonotope using Zonotopes reduction methods

        :param order: desired order
        :return: a Matrix zonotope
        """
        # Convert to zonotope and reduce
        reduced_zonotope = self.zonotope.reduce(order)

        # Convert zonotope back to matrix zonotope
        center = reduced_zonotope.center.reshape(self.center.shape)
        generators = np.zeros((reduced_zonotope.num_generators, center.shape[0], center.shape[1]))

        for i in range(reduced_zonotope.num_generators):
            generators[i] = reduced_zonotope.generators[:,i].reshape(center.shape)

        # Return new matrix zonotope
        return MatrixZonotope(center, generators)

    @property
    def max_norm(self) -> Tuple[float, np.ndarray]:
        """ Returns the maximum infinity norm on the set """
        return self.zonotope.max_norm
