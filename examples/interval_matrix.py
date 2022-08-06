import numpy as np
from interval import Interval

class IntervalMatrix(object):
    dim: int
    inf: np.ndarray
    sup: np.ndarray
    _interval: np.ndarray
    setting: str = 'sharpivmult'

    def __init__(self, matrix_center: np.ndarray, matrix_delta: np.ndarray, setting: str = None):
        self.dim = len(matrix_center)
        
        _matrix_delta = np.abs(matrix_delta)
        self.inf = matrix_center - _matrix_delta
        self.sup = matrix_center + _matrix_delta

        self._interval = Interval(self.inf, self.sup)
        
        if isinstance(setting, str):
            self.setting = setting

    def __str__(self):
        return f'Interval matrix: sup {self.sup}\ninf {self.inf}'

    @property
    def interval(self) -> Interval:
        """ Returns the interval representation """
        return self._interval

    def contains(self, X: np.ndarray) -> bool:
        """
        Returns true if the interval matrix contains X
        """
        return self.interval.contains(X)
