#########################################################################################
##
##                       DIFFERENTIAL-ALGEBRAIC EQUATION BLOCKS
##                                (pathsim/blocks/dae.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from scipy.linalg import lu_factor, lu_solve

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


class MassMatrixDAE(Block):
    """Differential-algebraic equation (DAE) system in mass-matrix form.

    Integrates an implicit system with a constant, possibly singular mass
    matrix :math:`M`

    .. math::

        M \\, \\dot{x} = \\mathrm{func}(x, u, t)


    where :math:`u` is the block input and the output is the full state
    :math:`y = x`.

    If :math:`M` is nonsingular the system is a plain (mass-weighted) ODE and
    the block presents the reduced right hand side :math:`\\dot{x} = M^{-1}
    \\mathrm{func}(x, u, t)` to the integration engine, reusing a single LU
    factorisation of :math:`M`.

    If :math:`M` is singular the all-zero rows are interpreted as algebraic
    constraints :math:`0 = \\mathrm{func}_a(x, u, t)`, and the corresponding
    states are the algebraic variables. They are eliminated at every evaluation
    by an internal Newton-Anderson iteration (:class:`.NewtonAnderson`), so the
    differential states integrate with any solver, explicit or implicit. The
    constraint is assumed to be index-1.

    Like the `ODE` block, the reduced right hand side is wrapped in a
    `DynamicOperator` and the engine Jacobian is taken from `op_dyn.jac_x`.

    Note
    ----
    For the singular case the mass matrix has to be in index-1 form, i.e. the
    differential rows must not weight the derivatives of the algebraic states
    (the corresponding block of :math:`M` is zero). For a nonsingular mass
    matrix an analytical reduced Jacobian :math:`M^{-1} \\partial \\mathrm{func}
    / \\partial x` is used when `jac` is supplied, otherwise the Jacobian comes
    from central finite differences through the reduced right hand side.

    Example
    -------
    A singular index-1 system where the second equation is an algebraic
    constraint :math:`x_0 + x_1 = u`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import MassMatrixDAE

        #M is singular -> second row is the algebraic constraint
        M = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

        def func(x, u, t):
            return np.array([
                -x[0] + x[1],        #x0' = -x0 + x1
                x[0] + x[1] - u[0]   #0 = x0 + x1 - u
                ])

        dae = MassMatrixDAE(func, M, initial_value=[0.0, 0.0])


    Parameters
    ----------
    func : callable
        right hand side function with signature `func(x, u, t)` returning an
        array of the dimension of the state `x`
    mass : array[array[float]]
        constant mass matrix `M`, possibly singular (all-zero rows mark
        algebraic constraints)
    initial_value : float, array[float]
        initial value / initial condition of the full state `x`
    jac : callable, None
        optional analytical jacobian of `func` with respect to `x` with
        signature `jac(x, u, t)`, central finite differences are used if `None`

    Attributes
    ----------
    mass : array[array[float]]
        the constant mass matrix `M`
    engine : Solver
        numerical integration engine for the differential states
    op_dyn : DynamicOperator
        dynamic operator wrapping the reduced right hand side
    opt : NewtonAnderson
        internal Newton-Anderson optimizer for the algebraic constraints
    """

    def __init__(self, func=lambda x, u, t: -x, mass=1.0, initial_value=0.0, jac=None):
        super().__init__()

        #right hand side and optional analytical jacobian
        self.func = func
        self.jac = jac

        #constant mass matrix
        M = np.atleast_2d(np.asarray(mass, dtype=float))
        n = M.shape[0]
        if M.shape != (n, n):
            raise ValueError(f"mass matrix must be square but has shape {M.shape}")
        self.mass = M

        #full initial state
        x0 = np.atleast_1d(initial_value).astype(float)
        if x0.size != n:
            raise ValueError(
                f"initial_value dimension {x0.size} does not match mass matrix {n}"
                )
        self._x0 = x0

        #partition into differential (nonzero row) and algebraic (zero row) states
        _nonzero_row = np.any(M != 0.0, axis=1)
        self._d = np.flatnonzero(_nonzero_row)
        self._a = np.flatnonzero(~_nonzero_row)

        #index-1 form: differential rows must not weight algebraic derivatives
        if self._a.size and np.any(M[np.ix_(self._d, self._a)] != 0.0):
            raise ValueError(
                "mass matrix is not in index-1 form: differential rows couple "
                "to the derivatives of algebraic states"
                )

        #LU factorisation of the (constant) differential mass block
        self._lu = lu_factor(M[np.ix_(self._d, self._d)])

        #the engine integrates only the differential states
        self.initial_value = x0[self._d]

        #initial guess and warm-start for the algebraic states
        self._xa = x0[self._a].copy()

        #internal optimizer for the algebraic constraints (consistent with engines)
        self.opt = NewtonAnderson()

        #dynamic operator for the reduced right hand side, mirrors the ODE block;
        #for a nonsingular mass matrix the reduced Jacobian M^-1 df/dx is analytic
        #when 'jac' is given, otherwise the operator falls back to finite differences
        _jac_x = None
        if self._a.size == 0 and jac is not None:
            _jac_x = lambda x, u, t: lu_solve(self._lu, np.atleast_2d(jac(x, u, t)))
        self.op_dyn = DynamicOperator(func=self._rhs, jac_x=_jac_x)

        #pre-size the output register to the full state
        self.outputs.update_from_array(x0)


    def __len__(self):
        #only the algebraic states introduce an input passthrough
        if not self._active:
            return 0
        return 1 if self._a.size else 0


    def reset(self):
        """Reset inputs, outputs, the engine and the warm-start of the
        algebraic states.
        """
        super().reset()
        self._xa = self._x0[self._a].copy()
        x_d = self.engine.state if self.engine else self.initial_value
        self.outputs.update_from_array(self._full(np.atleast_1d(x_d)))


    def _full(self, x_d):
        """Assemble the full state from the differential states and the current
        algebraic states.

        Parameters
        ----------
        x_d : array[float]
            differential states

        Returns
        -------
        x : array[float]
            full state with differential and algebraic components in place
        """
        x = np.empty(self.mass.shape[0])
        x[self._d] = x_d
        x[self._a] = self._xa
        return x


    def _solve_xa(self, x_d, u, t):
        """Eliminate the algebraic states by solving the zero-row constraints
        for the algebraic components, warm-started with the previous solution.
        Does not mutate the warm-start.

        Parameters
        ----------
        x_d : array[float]
            current differential states
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        xa : array[float]
            algebraic states satisfying the constraints
        """
        if self._a.size == 0:
            return self._xa

        def _res(xa):
            x = np.empty(self.mass.shape[0])
            x[self._d], x[self._a] = x_d, xa
            return self.func(x, u, t)[self._a]

        _jac = None
        if self.jac is not None:
            def _jac(xa):
                x = np.empty(self.mass.shape[0])
                x[self._d], x[self._a] = x_d, xa
                return np.atleast_2d(self.jac(x, u, t))[np.ix_(self._a, self._a)]

        xa, _, _ = solve_root(self.opt, _res, self._xa, _jac)
        return xa


    def _rhs(self, x_d, u, t):
        """Reduced right hand side of the differential states, with the
        algebraic states eliminated and the differential mass block inverted.

        Parameters
        ----------
        x_d : array[float]
            current differential states
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        dx_d : array[float]
            derivative of the differential states
        """
        if self._a.size == 0:
            x = self._full(x_d)
        else:
            xa = self._solve_xa(x_d, u, t)
            x = np.empty(self.mass.shape[0])
            x[self._d], x[self._a] = x_d, xa
        return lu_solve(self._lu, self.func(x, u, t)[self._d])


    def update(self, t):
        """Eliminate the algebraic states for the current input and expose the
        full state at the output.

        Parameters
        ----------
        t : float
            evaluation time
        """
        x_d, u = self.engine.state, self.inputs.to_array()
        self._xa = self._solve_xa(x_d, u, t)
        self.outputs.update_from_array(self._full(x_d))


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
        x_d, u = self.engine.state, self.inputs.to_array()

        #commit the warm-start at the current state, then linearize around it
        self._xa = self._solve_xa(x_d, u, t)
        f = lu_solve(self._lu, self.func(self._full(x_d), u, t)[self._d])
        J = self.op_dyn.jac_x(x_d, u, t)

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
        x_d, u = self.engine.state, self.inputs.to_array()
        self._xa = self._solve_xa(x_d, u, t)
        f = lu_solve(self._lu, self.func(self._full(x_d), u, t)[self._d])
        return self.engine.step(f, dt)


class FullyImplicitDAE(Block):
    """Fully-implicit differential-algebraic equation (DAE) system.

    Integrates a system given in fully-implicit residual form

    .. math::

        F(x, \\dot{x}, u, t) = 0


    where :math:`u` is the block input and the output is the state
    :math:`y = x`.

    At every evaluation the state derivative is recovered by an internal
    Newton-Anderson iteration (:class:`.NewtonAnderson`) that solves
    :math:`F(x, \\dot{x}, u, t) = 0` for :math:`\\dot{x}`, warm-started with the
    previous solution. The block then presents the explicit right hand side
    :math:`\\dot{x} = \\dot{x}(x, u, t)` to the integration engine, so the DAE
    integrates with any solver, explicit or implicit. The system is assumed to
    be index-1, i.e. the Jacobian :math:`\\partial F / \\partial \\dot{x}` is
    nonsingular.

    Like the `ODE` block, the recovered derivative is wrapped in a
    `DynamicOperator` and the engine Jacobian is taken from `op_dyn.jac_x`.

    Note
    ----
    The reduced Jacobian :math:`\\partial \\dot{x} / \\partial x` follows from
    the implicit function theorem,

    .. math::

        \\frac{\\partial \\dot{x}}{\\partial x}
        = -\\left(\\frac{\\partial F}{\\partial \\dot{x}}\\right)^{-1}
           \\frac{\\partial F}{\\partial x}

    It is used analytically when both `jac_x` and `jac_xdot` are supplied,
    otherwise it is approximated by finite differences through the recovered
    derivative. Supplying `jac_xdot` also accelerates the inner solve.

    Example
    -------
    A harmonic oscillator written in fully-implicit form, with state
    :math:`x = [\\text{position}, \\text{velocity}]`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import FullyImplicitDAE

        def func(x, xdot, u, t):
            return np.array([
                xdot[0] - x[1],   #position' = velocity
                xdot[1] + x[0]    #velocity' = -position
                ])

        dae = FullyImplicitDAE(func, initial_value=[1.0, 0.0])


    Parameters
    ----------
    func : callable
        residual function with signature `func(x, xdot, u, t)` returning an
        array of the dimension of the state `x`
    initial_value : float, array[float]
        initial value / initial condition of the state `x`
    jac_x : callable, None
        optional analytical jacobian of `func` with respect to `x` with
        signature `jac_x(x, xdot, u, t)`, used for the reduced jacobian
    jac_xdot : callable, None
        optional analytical jacobian of `func` with respect to `xdot` with
        signature `jac_xdot(x, xdot, u, t)`, accelerates the inner solve and is
        used for the reduced jacobian, central finite differences are used if
        `None`

    Attributes
    ----------
    engine : Solver
        numerical integration engine for the state `x`
    op_dyn : DynamicOperator
        dynamic operator wrapping the recovered derivative
    opt : NewtonAnderson
        internal Newton-Anderson optimizer for the state derivative
    """

    def __init__(
        self,
        func=lambda x, xdot, u, t: xdot + x,
        initial_value=0.0,
        jac_x=None,
        jac_xdot=None
        ):

        super().__init__()

        #implicit residual and optional analytical jacobians
        self.func = func
        self.jac_x = jac_x
        self.jac_xdot = jac_xdot

        #initial condition of the state (drives the engine)
        self.initial_value = np.atleast_1d(initial_value).astype(float)

        #warm-start for the recovered state derivative
        self._xdot = np.zeros_like(self.initial_value)

        #internal optimizer for the state derivative (consistent with engines)
        self.opt = NewtonAnderson()

        #dynamic operator for the recovered derivative, mirrors the ODE block;
        #the reduced Jacobian is analytic (implicit function theorem) when both
        #jacobians are given, otherwise the operator falls back to finite differences
        _jac_x = self._reduced_jac if (jac_x is not None and jac_xdot is not None) else None
        self.op_dyn = DynamicOperator(func=self._rhs, jac_x=_jac_x)

        #pre-size the output register to the state
        self.outputs.update_from_array(self.initial_value)


    def __len__(self):
        #the output is the integrated state, no direct input passthrough
        return 0


    def reset(self):
        """Reset inputs, outputs, the engine and the warm-start of the state
        derivative.
        """
        super().reset()
        self._xdot = np.zeros_like(self.initial_value)


    def _solve_xdot(self, x, u, t):
        """Recover the state derivative by solving the residual
        'func(x, xdot, u, t) = 0' for xdot, warm-started with the previous
        solution. Does not mutate the warm-start.

        Parameters
        ----------
        x : array[float]
            current state
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        xdot : array[float]
            state derivative satisfying the residual
        """
        _res = lambda xd: self.func(x, xd, u, t)
        _jac = None if self.jac_xdot is None \
            else (lambda xd: self.jac_xdot(x, xd, u, t))
        xdot, _, _ = solve_root(self.opt, _res, self._xdot, _jac)
        return xdot


    def _rhs(self, x, u, t):
        """Recovered explicit right hand side (the state derivative).

        Parameters
        ----------
        x : array[float]
            current state
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        xdot : array[float]
            recovered state derivative
        """
        return self._solve_xdot(x, u, t)


    def _reduced_jac(self, x, u, t):
        """Analytical reduced Jacobian from the implicit function theorem,
        :math:`-(\\partial F/\\partial \\dot{x})^{-1} \\partial F/\\partial x`.

        Parameters
        ----------
        x : array[float]
            current state
        u : array[float]
            current block input
        t : float
            evaluation time

        Returns
        -------
        J : array[array[float]]
            reduced jacobian of the recovered derivative
        """
        xdot = self._solve_xdot(x, u, t)
        Fx = np.atleast_2d(self.jac_x(x, xdot, u, t))
        Fxd = np.atleast_2d(self.jac_xdot(x, xdot, u, t))
        return -np.linalg.solve(Fxd, Fx)


    def update(self, t):
        """Expose the integrated state at the output.

        Parameters
        ----------
        t : float
            evaluation time
        """
        self.outputs.update_from_array(self.engine.state)


    def solve(self, t, dt):
        """Advance the implicit update equation of the solver with the recovered
        derivative and its Jacobian from the dynamic operator.

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
        self._xdot = self._solve_xdot(x, u, t)
        J = self.op_dyn.jac_x(x, u, t)

        return self.engine.solve(self._xdot, J, dt)


    def step(self, t, dt):
        """Compute the timestep update with the recovered derivative.

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
        self._xdot = self._solve_xdot(x, u, t)
        return self.engine.step(self._xdot, dt)
