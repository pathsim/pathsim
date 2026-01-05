#########################################################################################
##
##                               LOOKUP TABLE BLOCKS 
##                                (blocks/table.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from pathsim.blocks import Function
from scipy.interpolate import LinearNDInterpolator, interp1d
import numpy as np


# BLOCKS ================================================================================

class LUT(Function):
    """
    N-dimensional lookup table with linear interpolation functionality.

    This class implements a multi-dimensional lookup table that uses scipy's
    LinearNDInterpolator [#scipy]_ for piecewise linear interpolation in N-dimensional 
    space. The interpolation is based on Delaunay triangulation of the input points, 
    providing smooth linear interpolation between data points. For points outside 
    the convex hull of the input data, the interpolator returns NaN values.
    
    The LUT acts as a Function block.


    References
    ----------
    .. [#scipy] https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.LinearNDInterpolator.html


    Parameters
    ----------
    points : array_like of shape (n, ndim)
        2-D array of data point coordinates where n is the number of points
        and ndim is the dimensionality of the space. Each row represents a
        single data point in ndim-dimensional space.
    values : array_like of shape (n,) or (n, m)
        N-D array of data values at the corresponding points. If 1-D, represents
        scalar values at each point. If 2-D, each column represents a different
        output dimension (m output values per input point).
        

    Attributes
    ----------
    points : ndarray
        Stored array of input point coordinates.
    values : ndarray  
        Stored array of output values at each point.
    inter : scipy.interpolate.LinearNDInterpolator
        The scipy linear interpolator object used for interpolation.
    """

    def __init__(self, points=None, values=None):
        self.points = np.asarray(points) if points is not None else np.ones(2) 
        self.values = np.asarray(values) if values is not None else np.ones(2)

        self.inter = LinearNDInterpolator(self.points, self.values)

        super().__init__(func=lambda *x :self.inter(x))


class LUT1D(Function):
    """
    One-dimensional lookup table with linear interpolation functionality.
    
    This class implements a 1-dimensional lookup table that uses scipy's interp1d [#scipy]_
    for piecewise linear interpolation along a single axis. The interpolation
    provides linear interpolation between adjacent data points and supports
    extrapolation beyond the input data range using the 'extrapolate' fill mode.

    The LUT1D acts as a Function block.


    References
    ----------
    .. [#scipy] https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html


    Parameters
    ----------
    points : array_like of shape (n,)
        1-D array of monotonically increasing data point coordinates where n 
        is the number of points. These represent the independent variable values
        at which the dependent values are known.
    values : array_like of shape (n,) or (n, m)
        1-D or 2-D array of data values at the corresponding points. If 1-D,
        represents scalar values at each point. If 2-D with shape (n, m), 
        each column represents a different output dimension, allowing the
        lookup table to return m-dimensional vectors.
    fill_value : float or str, optional
        The value to use for points outside the interpolation range. If "extrapolate",
        the interpolator will use linear extrapolation. Default is "extrapolate".
        See https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.interp1d.html for more details


    Attributes
    ----------
    points : ndarray
        Flattened array of input point coordinates, stored as 1-D array.
    values : ndarray
        Stored array of output values at each point, preserving original shape.
    inter : scipy.interpolate.interp1d
        The scipy 1D interpolator object used for linear interpolation with
        extrapolation enabled beyond the data range.
    """

    def __init__(self, points=None, values=None, fill_value="extrapolate"):
        self.points = np.asarray(points).flatten() if points is not None else np.ones(2) 
        self.values = np.asarray(values) if values is not None else np.ones(2)
        
        # Handle both 1D and 2D values
        if self.values.ndim == 1:
            # Single output dimension
            self.inter = interp1d(self.points, self.values, fill_value=fill_value)
            def func(*x):
                return self.inter(x[0])
        else:
            # Multiple output dimensions - interpolate each column separately
            self.inter = interp1d(self.points, self.values, axis=0, fill_value=fill_value)
            def func(*x):
                return self.inter(x[0])

        super().__init__(func=func)