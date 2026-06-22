########################################################################################
##
##                  FUNCTIONS AND WRAPPERS FOR NUMERICAL DIFFERENTIATION 
##                                 (optim.numerical.py)
##
##                                   Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# NUMERICAL DIFFERENTIATION ============================================================

def num_jac(func, x, r=1e-3, tol=1e-8):
    """Numerically computes the jacobian of the function 'func'
    by central differences.

    The stepsize 'h' is adaptively computed as a relative perturbation
    'r' with an absolute floor 'tol'. The floor keeps the perturbation
    meaningful when components of 'x' are near zero, where a purely relative
    step would shrink below the floating point resolution of 'func' and the
    central difference would collapse to zero.

    Parameters
    ----------
    func : callable
        function to compute jacobian for
    x : float, array[float]
        value for function at which the jacobian is evaluated
    r : float
        relative perturbation
    tol : float
        absolute floor for the perturbation stepsize

    Returns
    -------
    jac : array[array[float]]
        2d jacobian array
    """

    #stepsize relative to value with an absolute floor
    H = np.clip(abs(r*x), tol, None)

    #catch scalar case (gradient)
    if np.isscalar(x):
        return 0.5 * (func(x + H) - func(x - H)) / H
    
    #perturbation matrix and jacobian
    return 0.5 * np.array(
        [(func(x + hv) - func(x - hv)) / h 
            for hv, h in zip(np.diag(H), H)]
        ).T


def num_autojac(func):
    """Wraps a function object such that it computes the jacobian 
    of the function with respect to the first argument.

    This is intended to compute the jacobian 'jac(x, u, t)' of 
    the right hand side function 'func(x, u, t)' of numerical 
    integrators with respect to 'x'.

    Parameters
    ----------
    func : callable
        function to wrap for jacobian 

    Returns
    -------
    wrap_func : callable
        wrapped funtion as numerical jacobian of 'func'
    """
    def wrap_func(*args):
        _x, *_args = args
        return num_jac(lambda x: func(x, *_args), _x)
    return wrap_func

