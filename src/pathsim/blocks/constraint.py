#########################################################################################
##
##                            ALGEBRAIC CONSTRAINT BLOCK
##                              (pathsim/blocks/constraint.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.newton import NewtonRaphson


# BLOCKS ================================================================================

class AlgebraicConstraint(Block):
    """Solve a nonlinear algebraic constraint for its internal unknown.

    At every evaluation the block solves the square nonlinear system

    .. math::

        \\mathrm{func}(x, u) = 0

    for the internal unknown :math:`x` given the current block input :math:`u`,
    and exposes the converged :math:`x` at its output:

    .. math::

        y = x \\quad\\text{such that}\\quad \\mathrm{func}(x, u) = 0


    The constraint is resolved with an internal damped Newton-Raphson iteration
    (:class:`.NewtonRaphson`) that is warm-started with the solution of the
    previous evaluation, so typically only a couple of iterations are required.
    If no analytical Jacobian :math:`\\partial \\mathrm{func} / \\partial x` is
    supplied it is approximated by central finite differences.

    This is the building block for steady-state operating points, implicit
    constitutive laws, chemical / phase equilibria and quasi-steady-state
    approximations, where an output is defined implicitly rather than
    explicitly. The input :math:`u` is wired dynamically through the block
    ports, so no input dimension has to be declared in advance.

    Note
    ----
    This block is purely algebraic. Its constraint solve is part of the
    algebraic component of the global system DAE and is therefore evaluated
    multiple times per timestep, each time `Simulation._update(t)` is called.
    The residual `func` must be purely algebraic and must not introduce states,
    delay or other dynamic behaviour.

    Example
    -------
    Compute the square root of the input as the positive root of
    :math:`x^2 - u = 0`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import AlgebraicConstraint

        #residual of the constraint x**2 - u = 0
        def func(x, u):
            return x**2 - u

        ac = AlgebraicConstraint(func, x0=1.0)


    A vector valued example, the equilibrium of a simple reversible reaction
    :math:`A \\rightleftharpoons B` with conservation :math:`x_A + x_B = u`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import AlgebraicConstraint

        k_f, k_r = 2.0, 1.0

        def func(x, u):
            return np.array([
                k_f*x[0] - k_r*x[1],   #reaction equilibrium
                x[0] + x[1] - u[0]     #mass conservation
                ])

        ac = AlgebraicConstraint(func, x0=[0.5, 0.5])


    Parameters
    ----------
    func : callable
        residual function of the constraint with signature `func(x, u)` that
        returns an array of the same dimension as `x`, where `x` is the
        internal unknown and `u` is the block input array
    x0 : float, array[float]
        initial value / initial guess for the internal unknown `x`
    jac : callable, None
        optional analytical jacobian of `func` with respect to `x` with
        signature `jac(x, u)`, central finite differences are used if `None`

    Attributes
    ----------
    solver : NewtonRaphson
        internal root solver for the algebraic constraint
    """

    def __init__(self, func=lambda x, u: x, x0=0.0, jac=None):
        super().__init__()

        #some checks to ensure that function works correctly
        if not callable(func):
            raise ValueError(f"'{func}' is not callable")

        #residual function and optional jacobian of the constraint
        self.func = func
        self.jac = jac

        #initial guess and warm-start for the internal unknown
        self.x0 = np.atleast_1d(x0).astype(float)
        self._x = self.x0.copy()

        #internal root solver for the constraint
        self.solver = NewtonRaphson()

        #pre-size the output register to the unknown dimension
        self.outputs.update_from_array(self._x)


    def __len__(self):
        return 1 if self._active else 0


    def reset(self):
        """Reset inputs, outputs and the warm-start of the internal unknown."""
        super().reset()
        self._x = self.x0.copy()
        self.outputs.update_from_array(self._x)


    def update(self, t):
        """Solve the algebraic constraint for the current input and set the
        output to the converged internal unknown.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #current block input
        u = self.inputs.to_array()

        #residual and optional jacobian as functions of x at fixed u
        _func = lambda x: self.func(x, u)
        _jac = None if self.jac is None else (lambda x: self.jac(x, u))

        #solve the constraint, warm-started with the previous solution
        self._x, _, _ = self.solver.solve(_func, self._x, _jac)

        #expose the converged unknown
        self.outputs.update_from_array(self._x)
