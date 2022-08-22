from .interval_matrix import IntervalMatrix
from .interval import Interval
from .matrix_zonotope import MatrixZonotope
from .zonotope import Zonotope
from .utils import concatenate_zonotope
from .reachability_analysis import compute_LTI_matrix_zonotope, compute_IO_LTI_matrix_zonotope, LTI_IO_reachability, LTI_reachability
from .cvx_zonotope import CVXZonotope

__author__ = 'Alessio Russo - alessior@kth.se'
__version__ = '0.0.4'
__url__ = 'https://github.com/rssalessio/Py-Data-Driven-Reachability-Analysis'
__info__ = {
    'version': __version__,
    'author': __author__,
    'url': __url__
}