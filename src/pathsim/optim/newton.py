########################################################################################
##
##                          NEWTON-RAPHSON ROOT SOLVER
##                                (optim/newton.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from .numerical import num_jac

from .._constants import (
    TOLERANCE,
    SOL_TOLERANCE_FPI,
    SOL_ITERATIONS_MAX
    )


# CLASS ================================================================================

class NewtonRaphson:
    """Damped Newton-Raphson solver for square nonlinear systems :math:`F(x) = 0`.

    Solves a system of nonlinear equations by the classic Newton iteration

    .. math::

        x_{k+1} = x_k - \\alpha_k \\, \\mathbf{J}(x_k)^{-1} F(x_k)

    where :math:`\\mathbf{J} = \\partial F / \\partial x` is the Jacobian and
    :math:`\\alpha_k \\in (0, 1]` is a damping factor selected by a backtracking
    line search on the residual norm :math:`\\| F \\|_\\infty`. The line search
    globalises the iteration so it converges from poor initial guesses; close
    to the solution the full step :math:`\\alpha_k = 1` is taken and quadratic
    convergence is recovered.

    The Jacobian is taken from a user supplied callable if available, otherwise
    it is approximated by central finite differences (:func:`num_jac`). Scalar
    and vector valued systems are handled uniformly by working with
    :func:`numpy.atleast_1d` / :func:`numpy.atleast_2d` internally.

    In PathSim this solver is the inner root finder for the algebraic and
    differential-algebraic blocks. It eliminates the algebraic variables of a
    semi-explicit DAE, recovers the slope of a fully-implicit or mass-matrix
    DAE and solves the constraint of an algebraic block. Each owning block keeps
    a single instance and warm-starts it with the previous solution by passing
    it as the initial value, so a handful of iterations per evaluation suffice.

    Example
    -------
    Solve the scalar equation :math:`x^2 - 2 = 0` for the positive root:

    .. code-block:: python

        import numpy as np
        from pathsim.optim.root import NewtonRaphson

        solver = NewtonRaphson()

        #residual and (optional) analytical jacobian
        def func(x):
            return x**2 - 2.0

        def jac(x):
            return 2.0 * x

        x, res, iterations = solver.solve(func, np.array([1.0]), jac)
        #x -> 1.41421356...

    Parameters
    ----------
    tolerance : float
        convergence tolerance on the residual norm :math:`\\| F \\|_\\infty`
    iterations_max : int
        maximum number of Newton iterations
    beta : float
        sufficient-decrease parameter for the backtracking line search
    line_search : bool
        if True, damp the Newton step with a backtracking line search,
        otherwise always take the full step
    fd_step : float
        absolute floor for the finite-difference Jacobian step, keeps the
        perturbation meaningful when components of the solution are near zero

    Attributes
    ----------
    iterations : int
        number of iterations used by the most recent solve
    residual : float
        residual norm reached by the most recent solve

    References
    ----------
    .. [1] Deuflhard, P. (2011). "Newton Methods for Nonlinear Problems:
           Affine Invariance and Adaptive Algorithms". Springer Series in
           Computational Mathematics, Vol. 35. :doi:`10.1007/978-3-642-23899-4`
    .. [2] Kelley, C. T. (1995). "Iterative Methods for Linear and Nonlinear
           Equations". SIAM, Frontiers in Applied Mathematics.
           :doi:`10.1137/1.9781611970944`
    """

    def __init__(
        self,
        tolerance=SOL_TOLERANCE_FPI,
        iterations_max=SOL_ITERATIONS_MAX,
        beta=1e-4,
        line_search=True,
        fd_step=1e-6
        ):

        #convergence tolerance on the residual norm
        self.tolerance = tolerance

        #maximum number of newton iterations
        self.iterations_max = iterations_max

        #sufficient-decrease parameter for the line search
        self.beta = beta

        #use backtracking line search to damp the newton step
        self.line_search = line_search

        #absolute floor for the finite-difference jacobian step
        self.fd_step = fd_step

        #diagnostics of the most recent solve
        self.iterations = 0
        self.residual = 0.0


    def __bool__(self):
        return True


    def _jacobian(self, func, x, jac):
        """Evaluate the Jacobian at 'x', from the user callable if provided,
        otherwise by central finite differences, as a 2d array.

        Parameters
        ----------
        func : callable
            residual function 'F(x)'
        x : array[float]
            point at which the jacobian is evaluated
        jac : callable | None
            optional analytical jacobian of 'func'

        Returns
        -------
        J : array[array[float]]
            2d jacobian matrix at 'x'
        """
        #central differences use a relative step with an absolute floor, so
        #the perturbation stays meaningful when components of 'x' are near zero
        _J = jac(x) if jac is not None else num_jac(func, x, tol=self.fd_step)
        return np.atleast_2d(_J)


    def solve(self, func, x0, jac=None):
        """Solve the nonlinear system 'func(x) = 0' starting from 'x0'.

        The initial value doubles as the warm-start, so passing the solution
        of the previous evaluation makes subsequent solves converge in very
        few iterations.

        Parameters
        ----------
        func : callable
            residual function 'F(x)' of the square system
        x0 : float, array[float]
            initial value / warm-start for the solution
        jac : callable | None
            optional analytical jacobian of 'func', central finite
            differences are used as a fallback if 'None'

        Returns
        -------
        x : array[float]
            converged solution
        res : float
            residual norm at the solution
        iterations : int
            number of newton iterations used
        """

        #working solution as float array (warm-start from 'x0')
        _x = np.atleast_1d(x0).astype(float).copy()

        #residual at the initial value
        _F = np.atleast_1d(func(_x)).astype(float)
        _res = np.linalg.norm(_F, np.inf)

        for i in range(self.iterations_max):

            #converged -> early exit
            if _res < self.tolerance:
                break

            #jacobian and newton direction (least squares fallback if singular)
            _J = self._jacobian(func, _x, jac)
            try:
                _dx = np.linalg.solve(_J, -_F)
            except np.linalg.LinAlgError:
                _dx, *_ = np.linalg.lstsq(_J, -_F, rcond=None)

            #backtracking line search on the residual norm
            _alpha = 1.0
            while True:
                _x_new = _x + _alpha * _dx
                _F = np.atleast_1d(func(_x_new)).astype(float)
                _res_new = np.linalg.norm(_F, np.inf)

                #sufficient decrease, full step or step too small -> accept
                if (not self.line_search
                    or _res_new < (1.0 - self.beta * _alpha) * _res
                    or _alpha < TOLERANCE):
                    break

                #shrink the step and retry
                _alpha *= 0.5

            #accept the step
            _x, _res = _x_new, _res_new

        #store diagnostics of the solve
        self.iterations, self.residual = i, _res

        return _x, _res, i
