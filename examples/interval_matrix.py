import numpy as np
from interval import Interval

class InteralMatrix(object):
    dim: int
    inf: np.ndarray
    sup: np.ndarray
    interval: np.ndarray
    setting: str = 'sharpivmult'

    def __init__(self, matrix_center: np.ndarray, matrix_delta: np.ndarray, setting: str = None):
        self.dim = len(matrix_center)
        
        _matrix_delta = np.abs(matrix_delta)
        self.inf = matrix_center - _matrix_delta
        self.sup = matrix_center + _matrix_delta

        self.interval = Interval(self.inf, self.sup)
        
        if isinstance(setting, str):
            self.setting = setting

    def __str__(self):
        return f'Interval matrix'

