########################################################################################
##
##                               ANDERSON ACCELERATION 
##                                (optim/anderson.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque

from .._constants import (
    TOLERANCE,
    OPT_RESTART,
    OPT_HISTORY
    )


# CLASS ================================================================================

class Anderson:
    """Anderson acceleration for fixed-point iteration.

    Solves nonlinear equations in fixed-point form :math:`x = g(x)` by
    computing the next iterate as a linear combination of previous iterates
    whose coefficients minimise the least-squares residual.

    .. math::

        x_{k+1} = \\sum_{i=0}^{m_k} \\alpha_i^{(k)}\\, g(x_{k-m_k+i})
        \\quad\\text{with}\\quad
        \\alpha^{(k)} = \\arg\\min \\bigl\\|\\sum_i \\alpha_i\\, r_{k-m_k+i}\\bigr\\|

    where :math:`r_k = g(x_k) - x_k` and :math:`m_k \\le m` is the current
    buffer depth.

    In PathSim this class is the inner fixed-point solver used by the
    simulation engine to resolve algebraic loops (cycles in the block
    diagram). Each loop-closing ``ConnectionBooster`` owns an ``Anderson``
    instance that accelerates convergence of the fixed-point iteration
    over the loop. The buffer depth ``m`` controls how many previous
    iterates are retained; larger values improve convergence on difficult
    loops at the cost of a small least-squares solve per iteration.

    Parameters
    ----------
    m : int
        buffer depth (number of stored iterates)
    restart : bool
        if True, clear the buffer once it reaches depth ``m``

    References
    ----------
    .. [1] Anderson, D. G. (1965). "Iterative Procedures for Nonlinear
           Integral Equations". Journal of the ACM, 12(4), 547--560.
           :doi:`10.1145/321296.321305`
    .. [2] Walker, H. F., & Ni, P. (2011). "Anderson Acceleration for
           Fixed-Point Iterations". SIAM Journal on Numerical Analysis,
           49(4), 1715--1735. :doi:`10.1137/10078356X`
    """

    def __init__(self, m=OPT_HISTORY, restart=OPT_RESTART):

        #length of buffer for next estimate
        self.m = m

        #restart after buffer length is reached?
        self.restart = restart

        #rolling difference buffers
        self.dx_buffer = deque(maxlen=self.m)
        self.dr_buffer = deque(maxlen=self.m)

        #prvious values
        self.x_prev = None
        self.r_prev = None


    def __bool__(self):
        return True


    def __len__(self):
        return len(self.dx_buffer[0]) if self.dx_buffer else 0


    def solve(self, func, x0, iterations_max=100, tolerance=1e-6):
        """Solve the function 'func' with initial 
        value 'x0' up to a certain tolerance.

        Note
        ----
        This method is for testing purposes only and 
        not used in the simulation loop.
        
        Parameters
        ----------
        func : callable
            function to solve
        x0 : numeric
            starting value for solution
        iterations_max : int
            maximum number of solver iterations
        tolerance : float
            convergence condition

        Returns
        -------
        x : numeric
            solution
        res : float
            residual
        i : int
            iteration count
        """

        _x = x0.copy()
        for i in range(iterations_max):
            _x, _res = self.step(_x, func(_x)+_x)
            if _res < tolerance:
                return _x, _res, i

        raise RuntimeError(f"did not converge in {iterations_max} steps")


    def reset(self):
        """reset the anderson accelerator"""

        #clear difference buffers
        self.dx_buffer.clear()
        self.dr_buffer.clear()

        #clear previous values
        self.x_prev = None
        self.r_prev = None


    def step(self, x, g):
        """Perform one iteration on the fixed-point solution.
    
        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        
        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm, fixed point error
        """

        #make numeric if value
        _x = np.asarray(x).flatten()
        _g = np.asarray(g).flatten()

        #residual (this gets minimized)
        _res = _g - _x
        
        #fallback to regular fpi if 'm == 0'
        if self.m == 0:
            return _g, np.linalg.norm(_res)
    
        #if no buffer, regular fixed-point update
        if self.x_prev is None:

            #save values for next iteration
            self.x_prev = _x
            self.r_prev = _res

            return _g, np.linalg.norm(_res)

        #append to difference buffer
        self.dx_buffer.append(_x - self.x_prev)
        self.dr_buffer.append(_res - self.r_prev)
        
        #save values for next iteration
        self.x_prev = _x
        self.r_prev = _res

        #if buffer size 'm' reached, restart
        if self.restart and len(self.dx_buffer) >= self.m:
            self.reset()
            return _g, np.linalg.norm(_res)

        #get difference matrices 
        dX = np.vstack(self.dx_buffer)
        dR = np.vstack(self.dr_buffer)

        #exit for scalar values (size-1 arrays after flatten)
        if _res.size == 1:

            #flatten to 1D for dot products
            dR_flat = dR.flatten()
            dX_flat = dX.flatten()

            #delta squared norm
            dR2 = np.dot(dR_flat, dR_flat)

            #catch division by zero
            if dR2 <= TOLERANCE:
                return _g, abs(_res[0])

            #new solution and residual
            return _x - _res[0] * np.dot(dR_flat, dX_flat) / dR2, abs(_res[0])

        #compute coefficients from least squares problem
        C, *_ = np.linalg.lstsq(dR.T, _res, rcond=None)

        #new solution and residual norm
        return _x - C @ dX, np.linalg.norm(_res)



class NewtonAnderson(Anderson):
    """Hybrid Newton--Anderson fixed-point solver.

    Extends :class:`Anderson` by prepending a Newton step when a Jacobian
    of :math:`g` is available.  The Newton step

    .. math::

        \\tilde{x} = x - (J_g - I)^{-1}\\,(g(x) - x)

    provides a quadratically convergent initial correction; the subsequent
    Anderson mixing step then stabilises the iteration and damps
    oscillations.

    In PathSim this solver is used inside every implicit ODE integration
    engine (BDF, DIRK, ESDIRK).  When a block provides a local Jacobian
    (e.g. ``ODE`` or ``LTI`` blocks), the Newton pre-step yields much
    faster convergence of the implicit update equation, reducing the
    number of fixed-point iterations per timestep.  Without a Jacobian the
    solver falls back to pure Anderson acceleration.

    References
    ----------
    .. [1] Anderson, D. G. (1965). "Iterative Procedures for Nonlinear
           Integral Equations". Journal of the ACM, 12(4), 547--560.
           :doi:`10.1145/321296.321305`
    .. [2] Walker, H. F., & Ni, P. (2011). "Anderson Acceleration for
           Fixed-Point Iterations". SIAM Journal on Numerical Analysis,
           49(4), 1715--1735. :doi:`10.1137/10078356X`
    """


    def solve(self, func, x0, jac=None, iterations_max=100, tolerance=1e-6):
        """Solve the function 'func' with initial value 
        'x0' up to a certain tolerance.

        Parameters
        ----------
        func : callable
            function to solve
        x0 : numeric
            starting value for solution
        jac : callable
            jacobian of 'func'
        iterations_max : int
            maximum number of solver iterations
        tolerance : float
            convergence condition

        Note
        ----
        This method is for testing purposes only and 
        not used in the simulation loop.

        Returns
        -------
        x : numeric
            solution
        res : float
            residual
        i : int
            iteration count
        """

        _x = x0.copy()
        for i in range(iterations_max):
            _x, _res = self.step(_x, func(_x)+_x, None if jac is None else jac(_x))
            if _res < tolerance:
                return _x, _res, i

        raise RuntimeError(f"did not converge in {iterations_max} steps")


    def _newton(self, x, g, jac):
        """Newton step on solution, where 'f=g-x' is the 
        residual and 'jac' is the jacobian of 'g'.

        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        jac : array
            evaluation of jacobian of 'g'

        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm
        """

        #preprocess formats
        _x = np.asarray(x).flatten()
        _g = np.asarray(g).flatten()

        _jac = np.asarray(jac)

        #compute residual
        _res = _g - _x

        #early exit for scalar or purely vectorial values
        if _res.size == 1 or np.ndim(_jac) == 1:
            
            return _x - _res / (_jac - 1.0), np.linalg.norm(_res)

        #vectorial values (newton raphson)
        return _x - np.linalg.solve(_jac - np.eye(len(_res)), _res), np.linalg.norm(_res)


    def step(self, x, g, jac=None):
        """Perform one iteration on the fixed-point solution. 
        
        If the jacobian of g 'jac' is provided, a newton step 
        is performed previous to anderson acceleration.
            
        Parameters
        ----------
        x : float, array
            current solution
        g : float, array
            current evaluation of g(x)
        jac : array
            evaluation of jacobian of 'g'

        Returns
        -------
        x : float, array
            new solution
        res : float
            residual norm
        """

        #newton step if jacobian available
        if jac is None: 

            #regular anderson step with residual
            return super().step(x, g)
        else: 
            #newton step with residual
            _x, res_norm = self._newton(x, g, jac)

            #anderson step with no residual
            y, _ = super().step(_x, g)

            return y, res_norm
