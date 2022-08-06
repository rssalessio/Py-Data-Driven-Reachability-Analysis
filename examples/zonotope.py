from __future__ import annotations
from ctypes import Union
import numpy as np
from copy import deepcopy

from interval import Interval

class Zonotope(object):
    """
    zonotope - object constructor for zonotope objects
    See also CORA library (https://github.com/TUMcps/CORA/blob/master/contSet/%40zonotope/zonotope.m)
    
    Description:
        This class represents zonotopes objects defined as
        {c + \sum_{i=1}^p beta_i * g^(i) | beta_i \in [-1,1]}.
    
    """
    Z: np.ndarray
    half_space: np.ndarray

    def __init__(self, center: np.ndarray, generator: np.ndarray):
        center = np.array(center)
        generator = np.array(generator)
        assert center.shape[0] == generator.shape[0], 'Center and generator do not have the same number of rows'
        self.Z = np.hstack([center, generator])
        self.half_space = np.array([])
    
    @property
    def center(self) -> np.ndarray:
        """ Returns the center of the Zonotope of dimension n"""
        return self.Z[:, :1].flatten()

    @property
    def generators(self) -> np.ndarray:
        """
        Returns the generators of the zonotope (matrix of dimension n x g),
        where n is the dimensionality and g is the number of generators.
        """
        return self.Z[:, 1:]

    @property
    def num_generators(self) -> np.ndarray:
        """ Number of generators """
        return self.generators.shape[1]

    @property
    def dimension(self) -> int:
        """ Dimensionality """
        return self.generators.shape[0]

    def copy(self) -> Zonotope:
        """ Returns a copy of the zonotope """
        return Zonotope(deepcopy(self.center)[:, np.newaxis], deepcopy(self.generators))

    def __add__(self, operand: Zonotope) -> Zonotope:
        ret = self.copy()
        if isinstance(operand, Zonotope):
            assert np.all(operand.dimension == self.dimension), \
                f"Operand has not the same dimension, {self.dimension} != {operand.dimension}"
            ret.Z[:, :1] += operand.center
            ret.Z = np.hstack([self.Z, operand.generators])
        else:
            raise Exception(f"Addition not implemented for type {type(operand)}")
        return ret

    def __mul__(self, operand: Union[int, float, np.ndarray]) -> Zonotope:
        ret = self.copy()
        if isinstance(operand, float) or isinstance(operand, int):
            ret.Z *= operand
        elif isinstance(operand, np.ndarray):
            assert operand.shape[1] == self.Z.shape[0]
            ret.Z = operand @ self.Z
        else:
            raise Exception(f"Multiplication not implemented for type {type(operand)}")
        return ret

    __rmul__ = __mul__   # commutative operation
    __matmul__ = __mul__

    def __str__(self):
        return f'Center: {self.center} - Generators: {self.generators.T}'

    def sample(self, batch_size: int = 1) -> np.ndarray:
        """
        Generates a uniform random points within a zonotope

        Return a matrix of size (b x n), where n is the
        dimensionality and b is the batch size

        :param batch_size: number of random points
        :return: A matrix where each row is a point of dimension n
        """
            
        beta = np.random.uniform(low=-1, high=1, size=(self.num_generators, batch_size))
        return (self.center[:, None] + np.dot(self.generators, beta)).T

    @property
    def interval(self) -> Interval:
        res = self.copy()
        center = res.center
        delta = np.abs(res.generators).sum(axis=1)
        return Interval(center - delta, center + delta)

    def contains(self, X: np.ndarray) -> bool:
        """
        Returns true if the zonotope contains X
        """
        return self.interval.contains(X)
            

c = np.ones((10, 1))
g = 5e-3 * np.ones((10, 1))
W = Zonotope(c, g)

Z = W * 10.0
# print(W)
# print(Z)

x = Z.sample(2)
#print(x.shape)
