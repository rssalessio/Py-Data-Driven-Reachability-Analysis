
import numpy as np
from typing import Union

class Interval(object):
    left_limit: np.ndarray
    right_limit: np.ndarray

    def __init__(self, left_limit: Union[float, np.ndarray], right_limit: Union[float, np.ndarray]):
        self.left_limit = np.array(left_limit)
        self.right_limit = np.array(right_limit)

        if self.left_limit.shape != self.right_limit.shape:
            raise ValueError('Left limit and right limit need to have the same shape')
        
        if np.any(self.left_limit >  self.right_limit):
            raise ValueError('Left limit needs cannot be greater than right limit')

    def __str__(self):
        return f'Interval: {self.left_limit}\n{self.right_limit}'


    def contains(self, X: np.ndarray) -> bool:
        """
        Returns true if interval contains X
        """
        assert isinstance(X, np.ndarray), "X is not a numpy array"
        assert X.shape == self.left_limit.shape, "X has not the correct shape"
        return np.all(X >= self.left_limit) and np.all(X <= self.right_limit)


