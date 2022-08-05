from __future__ import annotations
from email import generator
import numpy as np

class MatrixZonotope(object):
    """
    MatZonotope class. See also CORA Library
    https://github.com/TUMcps/CORA/blob/master/matrixSet/%40matZonotope/matZonotope.m
    """
    dims: int = 1
    gens: int = 0
    center: np.ndarray
    generators: np.ndarray

    def __init__(self, center: np.ndarray, generators: np.ndarray):
        assert len(center.shape) == 2 and len(generators.shape) == 2, \
            "Center and generators must be matrices"
        assert generators.shape[0] == center.shape[0], \
            "The number of rows for generator and cneter must be the same"
        self.center = center
        self.generators = generators

        self.dim = len(self.center.shape)
        self.gens = len(self.generators.shape)

    def __add__(self, operand: np.ndarray) -> MatrixZonotope:
        if(isinstance(operand, np.ndarray)):
            Z = MatrixZonotope(self.center, self.generators)
            Z.center = Z.center + operand
            return Z
        else:
            raise NotImplementedError

    __radd__ = __add__

