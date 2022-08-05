import numpy as np

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
        assert center.shape[0] == generator.shape[0], 'Center and generator do not have the same number of rows'
        self.Z = np.hstack([center, generator])
        self.half_space = np.array([])
    
    @property
    def center(self) -> np.ndarray:
        return self.Z[:, :1].flatten()

    @property
    def generators(self) -> np.ndarray:
        return self.Z[:, 1:]

    @property
    def num_generators(self) -> np.ndarray:
        return self.generators.shape[1]

    @property
    def dim(self) -> int:
        return len(self.center)

c = np.ones((10, 1))
g = 5e-3 * np.ones((10, 1))
W = Zonotope(c, g)
