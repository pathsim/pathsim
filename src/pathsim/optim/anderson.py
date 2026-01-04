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
    """Class for accelerated fixed-point iteration through anderson acceleration. 
    Solves a nonlinear set of equations given in the fixed-point form:

        x = g(x)

    Anderson Accelerstion tracks the evolution of the solution from the previous 
    iterations. The next step in the iteration is computed as a linear combination 
    of the previous iterates. The coefficients are computed to minimize the least 
    squares error of the fixed-point problem.

    Parameters
    ----------
    m : int
        buffer length
    restart : bool
        clear buffer when full
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
    """Modified class for hybrid anderson acceleration that can use a jacobian 'jac' of 
    the function 'g' for a newton step before the fixed point step for the initial 
    estimate before applying the anderson acceleration.

    If a jacobian 'jac' is available, this significantly improves the convergence 
    (speed and robustness) of the solution.
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
