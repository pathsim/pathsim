#########################################################################################
##
##                              ALGEBRAIC OPERATOR DEFINITION
##                                   (optim.operator.py)
##
##                                    Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import scipy.sparse as sp

from .numerical import num_jac
from .jacobian import NewtonOperator


# HELPERS ===============================================================================

def _apply(J, v):
    """Apply a jacobian to a vector, for dense arrays and scalars (via 'np.dot')
    but also for sparse matrices and matrix-free 'NewtonOperator' instances,
    where 'np.dot' does not do the right thing.

    Parameters
    ----------
    J : float, np.ndarray, scipy.sparse matrix, NewtonOperator
        jacobian
    v : np.ndarray
        vector the jacobian is applied to

    Returns
    -------
    Jv : np.ndarray
        jacobian-vector product
    """
    if sp.issparse(J) or isinstance(J, NewtonOperator):
        return J @ v
    return np.dot(J, v)


# OPERATOR CLASSES ======================================================================

class Operator(object):
    """Operator class for function evaluation and linearization.
    
    This class wraps a function to provide both direct evaluation and linear approximation
    capabilities. When linearized around a point x0, subsequent calls use the first-order
    Taylor approximation 
    
    .. math::

        f(x) \\approx f(x_0) + \\mathbf{J}(x_0) (x - x_0)


    instead of evaluating the function.
    
    The class supports multiple methods for Jacobian computation: user-provided analytical
    Jacobians, automatic differentiation via the Value class, and numerical differentiation
    as a fallback.
    
    Example
    -------
    Basic usage with automatic differentiation:
    
    .. code-block:: python
    
        def f(x):
            return x**2 + np.sin(x)
            
        op = Operator(f)
        
        # Direct function evaluation
        y1 = op(2.0)
        
        # Linearize at current point
        op.linearize(2.0)
        
        # Use linear approximation
        y2 = op(2.1)  # Returns f(2.0) + J(2.0) (2.1-2.0)
        
    With user-provided Jacobian:
    
    .. code-block:: python
    
        def f(x):
            return x**2 + np.sin(x)
            
        def df_dx(x):
            return 2*x + np.cos(x)
            
        op = Operator(f, jac=df_dx)
        
        op.linearize(2.0)  # Uses df_dx for Jacobian
    
    Parameters
    ----------
    func : callable
        The function to wrap
    jac : callable, optional
        Optional analytical Jacobian of func. If None, automatic or numerical
        differentiation will be used.

    Attributes
    ----------
    f0 : array_like
        function evaluation at operating point
    x0 : array_like
        operating point
    J : array_like
        jacobian matrix at operating point

    """

    def __init__(self, func, jac=None):
        self._func = func
        self._jac = jac
        self.f0 = None
        self.x0 = None
        self.J = None
        

    def __bool__(self):
        return True
        

    def __call__(self, x):
        """Evaluate the function or its linear approximation.
        
        If the operator has been linearized (f0 is not None), returns the linear
        approximation 
    
        .. math::

            f(x_0) + \\mathbf{J}(x_0) (x - x_0)
    

        otherwise, returns f(x) directly.
        
        Parameters
        ----------
        x : array_like
            Point at which to evaluate
            
        Returns
        -------
        value : array_like
            Function value or linear approximation
        """
        if self.f0 is None: 
            return self._func(x)
        dx = np.atleast_1d(x - self.x0)
        return self.f0 + _apply(self.J, dx)
        

    def jac(self, x):
        """Compute the Jacobian matrix at point x.
        
        Uses the following methods in order of preference:
        1. User-provided analytical Jacobian if available
        2. Automatic differentiation via Value class
        3. Numerical differentiation as fallback
        
        Parameters
        ----------
        x : array_like
            Point at which to evaluate the Jacobian
            
        Returns
        -------
        jacobian : ndarray
            Jacobian matrix at x
        """
        if self._jac is None:
            # Fallback to numerical differentiation
            return num_jac(self._func, x)
        else:
            # Use analytical jacobian
            return self._jac(x)
            

    def linearize(self, x):
        """Linearize the function at point x.
        
        Computes and stores both the function value and its Jacobian at x.
        After linearization, calls to the operator will use the linear
        approximation until reset() is called.
        
        Parameters
        ----------
        x : array_like
            Point at which to linearize the function
        """
        self.x0, self.f0, self.J = x, self._func(x), self.jac(x)
        

    def reset(self):
        """Reset the linearization.
        
        Clears the stored linearization point and Jacobian, causing the
        operator to evaluate the function directly on subsequent calls.
        """
        self.x0, self.f0, self.J = None, None, None


class DynamicOperator(object):
    """Operator class for dynamic system function evaluation and linearization.
    
    This class wraps a dynamic system function with signature f(x, u, t) to provide 
    both direct evaluation and linear approximation capabilities. When linearized 
    around operating points (x0, u0), subsequent calls use the first-order Taylor 
    approximation 
    
    .. math::

        f(x, u, t) \\approx f(x_0, u_0, t) + J_x(x_0, u_0, t) (x - x_0) + J_u(x_0, u_0, t) (u - u_0)

    instead of evaluating the function.
    
    The class supports multiple methods for Jacobian computation: user-provided analytical
    Jacobians, automatic differentiation via the Value class, and numerical differentiation
    as a fallback.
    
    Example
    -------
    Basic usage with automatic differentiation:
    
    .. code-block:: python
    
        def system(x, u, t):
            return -0.5*x + 2*u
            
        op = Operator(system)
        
        # Direct function evaluation
        y1 = op(x=1.0, u=0.5, t=0.0)
        
        # Linearize at current point
        op.linearize(x=1.0, u=0.5, t=0.0)
        
        # Use linear approximation
        y2 = op(x=1.1, u=0.6, t=0.1)
        
    With user-provided Jacobians:
    
    .. code-block:: python
    
        def system(x, u, t):
            return -0.5*x + 2*u
            
        def jac_x(x, u, t):
            return -0.5
            
        def jac_u(x, u, t):
            return 2.0
            
        op = Operator(system, jac_x=jac_x, jac_u=jac_u)
        
        op.linearize(x=1.0, u=0.5, t=0.0)
    
    Parameters
    ----------
    func : callable
        The function to wrap with signature func(x, u, t)
    jac_x : callable, optional
        Optional analytical Jacobian with respect to x. If None, automatic or 
        numerical differentiation will be used.
    jac_u : callable, optional
        Optional analytical Jacobian with respect to u. If None, automatic or 
        numerical differentiation will be used.

    Attributes
    ----------
    f0 : array_like
        Function evaluation at operating point
    x0 : array_like
        State operating point
    u0 : array_like
        Input operating point
    Jx : array_like
        Jacobian matrix with respect to x at operating point
    Ju : array_like
        Jacobian matrix with respect to u at operating point
    """

    def __init__(self, func, jac_x=None, jac_u=None):
        
        self._func = func
        
        self._jac_x = jac_x
        self._jac_u = jac_u

        self.f0 = None
        self.x0 = None
        self.u0 = None
        self.Jx = None
        self.Ju = None


    def __bool__(self):
        return True
        
        
    def __call__(self, x, u, t):
        """Evaluate the function or its linear approximation.
        
        If the operator has been linearized (f0 is not None), returns the linear
        approximation 
    
        .. math::

            f(x_0, u_0, t_0) + J_x(x_0, u_0, t_0) (x - x_0) + J_u(x_0, u_0, t_0) (u - u_0)
    
        otherwise, returns f(x, u, t) directly.
        
        Parameters
        ----------
        x : array_like
            State vector
        u : array_like
            Input vector
        t : float
            Time
            
        Returns
        -------
        value : array_like
            Function value or linear approximation
        """
        #no linearization available
        if self.f0 is None:
            return self._func(x, u, t)

        #linearization in x available
        if self.x0 is None: _fx = 0.0
        else: _fx = _apply(self.Jx, np.atleast_1d(x - self.x0))

        #linearization in u available
        if self.u0 is None: _fu = 0.0
        else: _fu = _apply(self.Ju, np.atleast_1d(u - self.u0))
        
        return self.f0 + _fx + _fu


    def jac_x(self, x, u, t):
        """Compute the Jacobian matrix with respect to x.
        
        Uses the following methods in order of preference:
        1. User-provided analytical Jacobian if available
        2. Automatic differentiation via Value class
        3. Numerical differentiation as fallback
        
        Parameters
        ----------
        x : array_like
            State vector
        u : array_like
            Input vector
        t : float
            Time
            
        Returns
        -------
        jacobian : ndarray
            Jacobian matrix with respect to x
        """
        if self._jac_x is None:
            # Keep u and t as is
            def func_x(_x):
                return self._func(_x, u, t)
            # Fallback to numerical differentiation
            return num_jac(func_x, x)
        else:
            # Use analytical jacobian
            return self._jac_x(x, u, t)
            

    def jac_u(self, x, u, t):
        """Compute the Jacobian matrix with respect to u.
        
        Uses the following methods in order of preference:
        1. User-provided analytical Jacobian if available
        2. Automatic differentiation via Value class
        3. Numerical differentiation as fallback
        
        Parameters
        ----------
        x : array_like
            State vector
        u : array_like
            Input vector
        t : float
            Time
            
        Returns
        -------
        jacobian : ndarray
            Jacobian matrix with respect to u
        """
        if self._jac_u is None:
            # Keep x and t as is
            def func_u(_u):
                return self._func(x, _u, t)
            # Fallback to numerical differentiation
            return num_jac(func_u, u)
        else:
            # Use analytical jacobian
            return self._jac_u(x, u, t)
        

    def linearize(self, x, u, t):
        """Linearize the function at point (x, u, t).
        
        Computes and stores the function value and Jacobians at the operating point.
        After linearization, calls to the operator will use the linear
        approximation until reset() is called.
        
        Parameters
        ----------
        x : array_like
            State vector
        u : array_like
            Input vector
        t : float
            Time
        """
        self.f0 = self._func(x, u, t)
        if x is not None:
            self.x0, self.Jx = np.atleast_1d(x), self.jac_x(x, u, t)
        if u is not None:
            self.u0, self.Ju = np.atleast_1d(u), self.jac_u(x, u, t)
        

    def reset(self):
        """Reset the linearization.
        
        Clears the stored linearization points and Jacobians, causing the
        operator to evaluate the function directly on subsequent calls.
        """
        self.f0 = None
        self.x0, self.Jx = None, None
        self.u0, self.Ju = None, None