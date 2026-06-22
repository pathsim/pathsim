#########################################################################################
##
##                       DIFFERENTIAL-ALGEBRAIC EQUATION BLOCKS
##                                (pathsim/blocks/dae.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from ._block import Block

from ..optim.operator import DynamicOperator
from ..optim.anderson import NewtonAnderson, solve_root


# BLOCKS ================================================================================

class SemiExplicitDAE(Block):
    """Semi-explicit index-1 differential-algebraic equation (DAE) system.

    Integrates a system that couples differential states :math:`x` to algebraic
    states :math:`z` through an explicit constraint

    .. math::

        \\begin{align}
            \\dot{x} &= f_\\mathrm{dyn}(x, z, u, t) \\\\
                   0 &= f_\\mathrm{alg}(x, z, u, t)
        \\end{align}


    where :math:`u` is the block input. The constraint is assumed to be
    index-1, i.e. the algebraic Jacobian :math:`\\partial f_\\mathrm{alg} /
    \\partial z` is nonsingular, so the algebraic states are locally a function
    :math:`z = z(x, u, t)` of the differential states.

    At every evaluation the algebraic states are eliminated by an internal
    Newton-Anderson iteration (:class:`.NewtonAnderson`) that solves
    :math:`f_\\mathrm{alg}(x, z, u, t) = 0` for :math:`z`, warm-started with the
    previous solution. The block then presents the reduced explicit right hand
    side

    .. math::

        \\dot{x} = f_\\mathrm{dyn}(x, z(x, u, t), u, t)

    to the integration engine, so the DAE integrates with any solver, explicit
    or implicit. The block output is the stacked state :math:`[x, z]`.

    Note
    ----
    The reduced Jacobian :math:`\\partial \\dot{x} / \\partial x` handed to
    implicit solvers is obtained from central finite differences through the
    eliminated constraint. An analytical Jacobian :math:`\\partial
    f_\\mathrm{alg} / \\partial z` accelerates the inner constraint solve and
    can be supplied via `jac_z`.

    Example
    -------
    A pendulum in index-1 form, where the angular velocity follows the
    differential equation and the tension is determined by the constraint:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import SemiExplicitDAE

        g, L = 9.81, 1.0

        #x = [theta, omega], z = [tension]
        def f_dyn(x, z, u, t):
            theta, omega = x
            return np.array([omega, -z[0]*np.sin(theta)])

        def f_alg(x, z, u, t):
            theta, omega = x
            return np.array([z[0] - (g*np.cos(theta) + L*omega**2)])

        dae = SemiExplicitDAE(f_dyn, f_alg, initial_value=[0.5, 0.0], z0=[1.0])


    Parameters
    ----------
    func_dyn : callable
        right hand side of the differential part with signature
        `func_dyn(x, z, u, t)` returning the derivative of `x`
    func_alg : callable
        residual of the algebraic constraint with signature
        `func_alg(x, z, u, t)` returning an array of the dimension of `z`
    initial_value : float, array[float]
        initial value / initial condition of the differential states `x`
    z0 : float, array[float]
        initial value / initial guess for the algebraic states `z`
    jac_z : callable, None
        optional analytical jacobian of `func_alg` with respect to `z` with
        signature `jac_z(x, z, u, t)`, central finite differences are used if
        `None`

    Attributes
    ----------
    engine : Solver
        numerical integration engine for the differential states `x`
    op_dyn : DynamicOperator
        dynamic operator wrapping the reduced right hand side, provides the
        engine Jacobian like the `ODE` block
    opt : NewtonAnderson
        internal Newton-Anderson optimizer for the algebraic constraint
    """

    def __init__(
        self,
        func_dyn=lambda x, z, u, t: -x,
        func_alg=lambda x, z, u, t: z,
        initial_value=0.0,
        z0=0.0,
        jac_z=None
        ):

        super().__init__()

        #differential and algebraic right hand side functions
        self.func_dyn = func_dyn
        self.func_alg = func_alg

        #optional analytical jacobian of the constraint w.r.t. z
        self.jac_z = jac_z

        #initial condition of the differential states (drives the engine)
        self.initial_value = np.atleast_1d(initial_value).astype(float)

        #initial guess and warm-start of the algebraic states
        self.z0 = np.atleast_1d(z0).astype(float)
        self._z = self.z0.copy()

        #internal optimizer for the algebraic constraint (consistent with engines)
        self.opt = NewtonAnderson()

        #dynamic operator for the reduced right hand side, mirrors the ODE block
        #and supplies the engine Jacobian through 'op_dyn.jac_x'
        self.op_dyn = DynamicOperator(func=self._rhs)

        #pre-size the output register to the stacked state [x, z]
        self.outputs.update_from_array(
            np.concatenate([self.initial_value, self.z0])
            )


    def __len__(self):
        #the algebraic states generally depend on the input, so the block
        #carries an algebraic passthrough through z
        return 1 if self._active else 0


    def reset(self):
        """Reset inputs, outputs, the engine and the warm-start of the
        algebraic states.
        """
        super().reset()
        self._z = self.z0.copy()
        x = self.engine.state if self.engine else self.initial_value
        self.outputs.update_from_array(
            np.concatenate([np.atleast_1d(x), self._z])
            )


    def _solve_z(self, x, u, t):
        """Eliminate the algebraic states by solving the constraint
        'func_alg(x, z, u, t) = 0' for z, warm-started with the previous
        solution. Does not mutate the warm-start.

        Parameters
        ----------
        x : array[float]
            current differential states
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        z : array[float]
            algebraic states satisfying the constraint
        """
        _func = lambda z: self.func_alg(x, z, u, t)
        _jac = None if self.jac_z is None else (lambda z: self.jac_z(x, z, u, t))
        z, _, _ = solve_root(self.opt, _func, self._z, _jac)
        return z


    def _rhs(self, x, u, t):
        """Reduced explicit right hand side with the algebraic states
        eliminated.

        Parameters
        ----------
        x : array[float]
            current differential states
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        dx : array[float]
            derivative of the differential states
        """
        return self.func_dyn(x, self._solve_z(x, u, t), u, t)


    def update(self, t):
        """Eliminate the algebraic states for the current input and expose the
        stacked state [x, z] at the output.

        Parameters
        ----------
        t : float
            evaluation time
        """
        x, u = self.engine.state, self.inputs.to_array()
        self._z = self._solve_z(x, u, t)
        self.outputs.update_from_array(
            np.concatenate([np.atleast_1d(x), self._z])
            )


    def solve(self, t, dt):
        """Advance the implicit update equation of the solver with the reduced
        right hand side and its Jacobian from the dynamic operator.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        -------
        error : float
            solver residual norm
        """
        x, u = self.engine.state, self.inputs.to_array()

        #commit the warm-start at the current state, then linearize around it
        self._z = self._solve_z(x, u, t)
        f = self.func_dyn(x, self._z, u, t)
        J = self.op_dyn.jac_x(x, u, t)

        return self.engine.solve(f, J, dt)


    def step(self, t, dt):
        """Compute the timestep update with the reduced right hand side.

        Parameters
        ----------
        t : float
            evaluation time
        dt : float
            integration timestep

        Returns
        -------
        success : bool
            step was successful
        error : float
            local truncation error from adaptive integrators
        scale : float
            timestep rescale from adaptive integrators
        """
        x, u = self.engine.state, self.inputs.to_array()
        self._z = self._solve_z(x, u, t)
        f = self.func_dyn(x, self._z, u, t)
        return self.engine.step(f, dt)
