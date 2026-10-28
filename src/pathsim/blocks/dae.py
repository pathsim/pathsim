#########################################################################################
##
##                       DIFFERENTIAL-ALGEBRAIC EQUATION BLOCKS
##                                (pathsim/blocks/dae.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import scipy.sparse as sp

from scipy.linalg import lu_factor, lu_solve

from ._block import Block

from ..utils.register import Register

from ..optim.operator import DynamicOperator
from ..optim.jacobian import NewtonOperator
from ..optim.anderson import NewtonAnderson, solve_root, _factorize_sparse

from .._constants import SOL_TOLERANCE_FPI


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
    or implicit. The block output is the stacked state :math:`[x, z]`, or
    :math:`y = f_\\mathrm{out}(x, z, u, t)` if an output function is supplied.
    Without algebraic states (empty `z0`) the block reduces to a plain ODE and
    the constraint solve is skipped.

    Note
    ----
    The reduced Jacobian :math:`\\partial \\dot{x} / \\partial x` handed to
    implicit solvers follows from the implicit function theorem,

    .. math::

        \\frac{\\partial \\dot{x}}{\\partial x}
        = \\frac{\\partial f_\\mathrm{dyn}}{\\partial x}
        - \\frac{\\partial f_\\mathrm{dyn}}{\\partial z}
          \\left(\\frac{\\partial f_\\mathrm{alg}}{\\partial z}\\right)^{-1}
          \\frac{\\partial f_\\mathrm{alg}}{\\partial x}

    It is used analytically when `jac_z`, `jac_dyn_x`, `jac_dyn_z` and
    `jac_alg_x` are all supplied, otherwise it is approximated by central
    finite differences through the eliminated constraint, which requires one
    constraint solve per differential state and becomes expensive for large
    systems. An analytical `jac_z` alone accelerates the inner constraint
    solve.

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
    jac_dyn_x : callable, None
        optional analytical jacobian of `func_dyn` with respect to `x` with
        signature `jac_dyn_x(x, z, u, t)`, used for the reduced jacobian
    jac_dyn_z : callable, None
        optional analytical jacobian of `func_dyn` with respect to `z` with
        signature `jac_dyn_z(x, z, u, t)`, used for the reduced jacobian
    jac_alg_x : callable, None
        optional analytical jacobian of `func_alg` with respect to `x` with
        signature `jac_alg_x(x, z, u, t)`, used for the reduced jacobian
    tolerance : float
        convergence tolerance on the residual norm of the algebraic constraint
        solve, loosen it for ill-conditioned constraints whose residual cannot
        reach the default due to roundoff
    func_out : callable, None
        optional output function with signature `func_out(x, z, u, t)`, the
        block output is the stacked state `[x, z]` if `None`

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
        jac_z=None,
        jac_dyn_x=None,
        jac_dyn_z=None,
        jac_alg_x=None,
        tolerance=SOL_TOLERANCE_FPI,
        func_out=None
        ):

        super().__init__()

        #differential and algebraic right hand side functions
        self.func_dyn = func_dyn
        self.func_alg = func_alg

        #optional output function, stacked state [x, z] if 'None'
        self.func_out = func_out

        #convergence tolerance of the algebraic constraint solve
        self.tolerance = tolerance

        #optional analytical jacobian of the constraint w.r.t. z
        self.jac_z = jac_z

        #optional analytical jacobians for the reduced jacobian
        self.jac_dyn_x = jac_dyn_x
        self.jac_dyn_z = jac_dyn_z
        self.jac_alg_x = jac_alg_x

        #initial condition of the differential states (drives the engine)
        self.initial_value = np.atleast_1d(initial_value).astype(float)

        #initial guess and warm-start of the algebraic states
        self.z0 = np.atleast_1d(z0).astype(float)
        self._z = self.z0.copy()

        #internal optimizer for the algebraic constraint (consistent with engines)
        self.opt = NewtonAnderson()

        #dynamic operator for the reduced right hand side, mirrors the ODE block
        #and supplies the engine Jacobian through 'op_dyn.jac_x'; the reduced
        #Jacobian is analytic (implicit function theorem) when all jacobians
        #are given, otherwise the operator falls back to finite differences
        _all_jacs = (jac_z, jac_dyn_x, jac_dyn_z, jac_alg_x)
        _jac_x = self._reduced_jac if all(j is not None for j in _all_jacs) else None
        self.op_dyn = DynamicOperator(func=self._rhs, jac_x=_jac_x)

        #pre-size the output register to the stacked state [x, z]
        if self.func_out is None:
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
        if self.func_out is None:
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
        #nothing to solve without algebraic states
        if self._z.size == 0:
            return self._z

        _func = lambda z: self.func_alg(x, z, u, t)
        _jac = None if self.jac_z is None else (lambda z: self.jac_z(x, z, u, t))
        z, _, _ = solve_root(self.opt, _func, self._z, _jac, tolerance=self.tolerance)
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


    def _reduced_jac(self, x, u, t):
        """Analytical reduced Jacobian from the implicit function theorem,
        :math:`f_x - f_z (g_z)^{-1} g_x` with the dynamic part `f` and the
        algebraic constraint `g`.

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
        J : array[array[float]]
            reduced jacobian of the differential states
        """
        z = self._solve_z(x, u, t)
        fx = np.atleast_2d(self.jac_dyn_x(x, z, u, t))
        if z.size == 0:
            return fx
        fz = np.atleast_2d(self.jac_dyn_z(x, z, u, t))
        gx = np.atleast_2d(self.jac_alg_x(x, z, u, t))
        gz = np.atleast_2d(self.jac_z(x, z, u, t))
        return fx - fz @ np.linalg.solve(gz, gx)


    def update(self, t):
        """Eliminate the algebraic states for the current input and expose the
        output function, or the stacked state [x, z], at the output.

        Parameters
        ----------
        t : float
            evaluation time
        """
        x, u = self.engine.state, self.inputs.to_array()
        self._z = self._solve_z(x, u, t)
        if self.func_out is None:
            y = np.concatenate([np.atleast_1d(x), self._z])
        else:
            y = self.func_out(x, self._z, u, t)
        self.outputs.update_from_array(y)


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
    tolerance : float
        convergence tolerance on the residual norm of the algebraic constraint
        solve, loosen it for ill-conditioned constraints whose residual cannot
        reach the default due to roundoff

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

    def __init__(
        self,
        func=lambda x, u, t: -x,
        mass=1.0,
        initial_value=0.0,
        jac=None,
        tolerance=SOL_TOLERANCE_FPI
        ):

        super().__init__()

        #right hand side and optional analytical jacobian
        self.func = func
        self.jac = jac

        #convergence tolerance of the algebraic constraint solve
        self.tolerance = tolerance

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

        #partition into differential (nonzero column) and algebraic (zero column) states
        _nonzero_column = np.any(M != 0.0, axis=0)
        self._d = np.flatnonzero(_nonzero_column)
        self._a = np.flatnonzero(~_nonzero_column)

        #partition into differential (nonzero row) and algebraic (zero row) equations
        _non_zero_rows = np.any(M != 0.0, axis=1)
        self._d_equations = np.flatnonzero(_non_zero_rows)
        self._a_equations = np.flatnonzero(~_non_zero_rows)

        #index-1 form: differential rows must not weight algebraic derivatives
        if self._a.size and np.any(M[np.ix_(self._d_equations, self._a)] != 0.0):
            raise ValueError(
                "mass matrix is not in index-1 form: differential rows couple "
                "to the derivatives of algebraic states"
                )

        #factorise the constant differential mass block when it is square; use
        #the pseudoinverse for rectangular reduced systems so underdetermined
        #differential blocks still admit a minimum-norm solution.
        self._d_mass = M[np.ix_(self._d_equations, self._d)]
        if self._d_mass.shape[0] == self._d_mass.shape[1]:
            self._lu = lu_factor(self._d_mass)
            self._solve_mass_rhs = lambda rhs: lu_solve(self._lu, rhs)
            self._solve_mass_matrix = lambda mat: lu_solve(self._lu, mat)
        else:
            # Use SVD once and apply the pseudoinverse implicitly for stability
            # and to avoid forming a dense pseudoinverse matrix.
            U, s, Vt = np.linalg.svd(self._d_mass, full_matrices=False)
            eps = np.finfo(float).eps
            tol = max(self._d_mass.shape) * (s.max() if s.size else 0.0) * eps
            s_inv = np.array([1.0/si if si > tol else 0.0 for si in s])

            # store factors (optional) and set solver callables
            self._svd_U = U
            self._svd_s_inv = s_inv
            self._svd_Vt = Vt

            def _solve_mass_rhs(rhs):
                # rhs can be vector or matrix; apply V * S_inv * (U.T @ rhs)
                return (Vt.T * s_inv) @ (U.T @ rhs)

            def _solve_mass_matrix(mat):
                # mat: matrix of columns to be solved
                return (Vt.T * s_inv) @ (U.T @ mat)

            self._solve_mass_rhs = _solve_mass_rhs
            self._solve_mass_matrix = _solve_mass_matrix

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
            def _jac_x(x, u, t):
                J = np.atleast_2d(jac(x, u, t))[np.ix_(self._d_equations, self._d)]
                return self._solve_mass_matrix(J)
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
            return self.func(x, u, t)[self._a_equations]

        _jac = None
        if self.jac is not None:
            def _jac(xa):
                x = np.empty(self.mass.shape[0])
                x[self._d], x[self._a] = x_d, xa
                return np.atleast_2d(self.jac(x, u, t))[np.ix_(self._a_equations, self._a)]

        xa, _, _ = solve_root(self.opt, _res, self._xa, _jac, tolerance=self.tolerance)
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
        return self._solve_mass_rhs(self.func(x, u, t)[self._d_equations])


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
        f = self._solve_mass_rhs(self.func(self._full(x_d), u, t)[self._d_equations])

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
        f = self._solve_mass_rhs(self.func(self._full(x_d), u, t)[self._d_equations])
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
    tolerance : float
        convergence tolerance on the residual norm of the state derivative
        solve, loosen it for ill-conditioned residuals that cannot reach the
        default due to roundoff

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
        jac_xdot=None,
        tolerance=SOL_TOLERANCE_FPI
        ):

        super().__init__()

        #implicit residual and optional analytical jacobians
        self.func = func
        self.jac_x = jac_x
        self.jac_xdot = jac_xdot

        #convergence tolerance of the state derivative solve
        self.tolerance = tolerance

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
        xdot, _, _ = solve_root(self.opt, _res, self._xdot, _jac, tolerance=self.tolerance)
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


class SparseLinearDAE(Block):
    """Large sparse linear differential-algebraic equation (DAE) system.

    Integrates the linear descriptor system

    .. math::

        \\begin{align}
            S_1 \\, x + S_2 \\, \\dot{x} &= B \\, u \\\\
                                       y &= C \\, x + D \\, u
        \\end{align}

    where :math:`S_1`, :math:`S_2`, :math:`B`, :math:`C` and :math:`D` are ``scipy.sparse``
    matrices and :math:`S_2` is generally singular, so the algebraic rows
    (rows without derivatives) are constraints.

    The system is never converted to a dense matrix. The nonzero rows
    :math:`S_{2,r}` of :math:`S_2` define the *differential states*
    :math:`d = S_{2,r}\\,x` (for a circuit these are the capacitor charges and
    the inductor fluxes) and with the embedding :math:`P` of these rows into the
    full system, :math:`S_2 = P\\,S_{2,r}`, the DAE becomes

    .. math::

        S_1\\,x + P\\,\\dot{d} = B\\,u, \\qquad S_{2,r}\\,x = d

    which is the *constant sparse linear system*

    .. math::

        \\begin{bmatrix} S_1 & P \\\\ S_{2,r} & 0 \\end{bmatrix}
        \\begin{bmatrix} x \\\\ \\dot{d} \\end{bmatrix}
        =
        \\begin{bmatrix} B\\,u \\\\ d \\end{bmatrix}

    that is factorized once with a sparse LU. Solving it yields the derivative
    :math:`\\dot{d}(d, u)` of the differential states and the full state
    :math:`x(d, u)`. So the block presents an ordinary explicit ODE in :math:`d`
    to the integration engine, which therefore works with any solver, explicit
    or implicit, including the adaptive ones, and the output
    :math:`y = C\\,x(d, u) + D\\,u` is evaluated algebraically.

    The Jacobian :math:`\\partial \\dot{d} / \\partial d` of this reduction is
    dense (a Schur complement) and is never formed. It is handed to the
    implicit engines as a matrix-free :class:`.NewtonOperator` that solves the
    Newton system :math:`(c J - I)\\,y = r` of the :class:`.NewtonAnderson`
    optimizer with one sparse factorization of

    .. math::

        \\begin{bmatrix} S_1 & P \\\\ S_{2,r} & -c\\,I \\end{bmatrix}

    per distinct scale :math:`c` (for a fixed timestep these repeat, so the
    factorizations are cached).

    Note
    ----
    The reduction requires an index-1 system, which means that the matrix above
    has to be nonsingular. For circuits this excludes capacitor loops (for
    example parallel capacitors, or a capacitor directly across an ideal voltage
    source) and inductor cut-sets (for example inductors in series without a
    resistive path), where the charges / fluxes are not independent.
    Constructing the block raises a ``ValueError`` in these cases.

    The differential states :math:`d` are scaled by the row norms of
    :math:`S_{2,r}`, so that they have the magnitude of the system variables
    (the rows of :math:`S_{2,r}` are normalized, :math:`P` carries the scale).
    This is an exact reformulation but matters numerically, since the
    tolerances of the implicit engines are absolute.

    The initial condition ``initial_value`` is the initial value of these
    differential states, zero means no stored charge / flux, which is also the
    initial condition that `ElecSolver` assumes. Use ``x0`` to specify the
    initial condition as a full state instead.

    Example
    -------
    An RC low-pass driven by a voltage source :math:`u`, with the unknowns
    :math:`x = [v_1, v_2, i]`:

    .. code-block:: python

        import numpy as np
        import scipy.sparse as sp
        from pathsim.blocks import SparseLinearDAE

        R, C = 1e3, 1e-6

        #v1 = u,  v2 - v1 + R i = 0,  C dv2/dt - i = 0
        S1 = sp.csr_matrix([[1., 0., 0.], [-1., 1., R], [0., 0., -1.]])
        S2 = sp.csr_matrix([[0., 0., 0.], [0., 0., 0.], [0., C, 0.]])
        B  = sp.csr_matrix([[1.], [0.], [0.]])

        rc = SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[0., 1., 0.]]))

    Parameters
    ----------
    S1 : scipy.sparse matrix, shape (n, n)
        matrix of the non-differentiated states
    S2 : scipy.sparse matrix, shape (n, n)
        matrix of the differentiated states, may be singular
    B : scipy.sparse matrix, shape (n, nu)
        input matrix, one column per block input. For circuits with several
        sources this is the superposition of the unit sources, :math:`B\\,u =
        \\sum_k u_k b_k`
    C : scipy.sparse matrix, shape (ny, n), None
        output matrix, the full state :math:`y = x` is the output if `None`
    D : scipy.sparse matrix, shape (ny, nu), None
        direct feedthrough matrix from the inputs to the outputs, zero if `None`
    initial_value : array[float], None
        initial value of the differential states :math:`d`, zeros if `None`
    x0 : array[float], None
        initial condition as a full state, converted to :math:`d = S_{2,r}\\,x_0`,
        takes precedence over `initial_value`
    input_labels : list[str], None
        port aliases for the inputs
    output_labels : list[str], None
        port aliases for the outputs

    Attributes
    ----------
    engine : Solver
        numerical integration engine for the differential states :math:`d`
    op_dyn : DynamicOperator
        dynamic operator of the reduced ODE :math:`\\dot{d}(d, u)`
    op_alg : DynamicOperator
        dynamic operator of the output :math:`y(d, u)`
    n : int
        number of states of the full system
    nd : int
        number of differential states
    """

    #number of cached factorizations of the Newton system (distinct scales 'c')
    _NEWTON_CACHE = 8

    def __init__(
        self,
        S1,
        S2,
        B,
        C=None,
        D=None,
        initial_value=None,
        x0=None,
        input_labels=None,
        output_labels=None
        ):

        super().__init__()

        #system matrices
        S1 = sp.csc_matrix(S1, dtype=float)
        S2 = sp.csr_matrix(S2, dtype=float)
        S2.eliminate_zeros()
        n = S1.shape[0]

        if S1.shape != (n, n) or S2.shape != (n, n):
            raise ValueError(
                f"S1 and S2 must be square and equally sized but are {S1.shape}, {S2.shape}"
                )

        B = sp.csr_matrix(B, dtype=float)
        if B.shape[0] != n:
            raise ValueError(f"B must have {n} rows but has {B.shape[0]}")

        C = sp.identity(n, format="csr") if C is None else sp.csr_matrix(C, dtype=float)
        if C.shape[1] != n:
            raise ValueError(f"C must have {n} columns but has {C.shape[1]}")

        self.n, self.nu, self.ny = n, B.shape[1], C.shape[0]

        D = sp.csr_matrix((self.ny, self.nu)) if D is None else sp.csr_matrix(D, dtype=float)
        if D.shape != (self.ny, self.nu):
            raise ValueError(f"D must have shape {(self.ny, self.nu)} but has {D.shape}")

        self.S1, self.S2, self.B, self.C, self.D = S1, S2, B, C, D

        #differential states d = S2r x from the nonzero rows of S2
        rows = np.flatnonzero(np.diff(S2.indptr) > 0)
        nd = len(rows)
        if nd == 0:
            raise ValueError("S2 has no nonzero rows, the system has no dynamics")

        self.nd = nd

        #scale the differential states by the row norms of S2r, so they have the
        #magnitude of the system variables (e.g. volts and amps instead of coulombs
        #and webers, which can be many orders smaller). The optimizer tolerances of
        #the implicit engines are absolute, so badly scaled states converge too early.
        #This is an exact reformulation, S2 = P0 S2r = (P0 Dn) (Dn^-1 S2r)
        S2r = S2[rows]
        self._dscale = np.sqrt(np.asarray(S2r.multiply(S2r).sum(axis=1)).ravel())
        self._S2r = sp.diags(1.0 / self._dscale) @ S2r
        self._P = sp.csr_matrix((self._dscale, (rows, np.arange(nd))), shape=(n, nd))

        #constant sparse system [[S1, P], [S2r, 0]] [x, dd] = [B u, d]
        K = sp.bmat([[S1, self._P], [self._S2r, None]], format="csc")
        self._lu = _factorize_sparse(K)
        if self._lu is None:
            raise ValueError(
                "the DAE is not index-1: the matrix [[S1, P], [S2r, 0]] is singular. "
                "For circuits: capacitor loops (e.g. parallel capacitors or a capacitor "
                "across an ideal voltage source) or inductor cut-sets (e.g. series "
                "inductors without a resistive path)."
                )

        #factorizations of the newton system for different scales 'c'
        self._lu_newton = {}

        #input dependence of the derivative and the output (dense, small: nu columns)
        _rhs = np.zeros((n + nd, self.nu))
        _rhs[:n] = B.toarray()
        _sol = self._lu.solve(_rhs) if self.nu else np.zeros((n + nd, 0))
        self._jac_u_dyn = _sol[n:]
        self._jac_u_alg = C @ _sol[:n] + D.toarray()
        self._feedthrough = bool(np.any(self._jac_u_alg != 0.0))

        #lazily evaluated dense jacobian of the output w.r.t. the states
        self._jac_x_alg = None

        #last solution of the sparse system, shared by dynamics and output
        self._cache = None

        #initial condition of the differential states
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).ravel()
            if x0.size != n:
                raise ValueError(f"x0 must have {n} entries but has {x0.size}")
            initial_value = self._S2r @ x0
        self.initial_value = np.zeros(nd) if initial_value is None \
            else np.atleast_1d(initial_value).astype(float)
        if self.initial_value.size != nd:
            raise ValueError(
                f"initial_value must have {nd} entries (differential states) "
                f"but has {self.initial_value.size}"
                )

        #matrix-free jacobian of the reduced ODE for the implicit engines
        self._jac = NewtonOperator(
            newton_solve=self._newton_solve,
            matvec=self._matvec,
            shape=(nd, nd)
            )

        #operators for the reduced dynamics and the output
        self.op_dyn = DynamicOperator(
            func=self._func_dyn,
            jac_x=lambda x, u, t: self._jac,
            jac_u=lambda x, u, t: self._jac_u_dyn
            )
        self.op_alg = DynamicOperator(
            func=self._func_alg,
            jac_x=self._jac_alg_x,
            jac_u=lambda x, u, t: self._jac_u_alg
            )

        #pre-sized, labeled ports
        self.inputs = Register(
            size=self.nu,
            mapping=None if input_labels is None else {k: i for i, k in enumerate(input_labels)}
            )
        self.outputs = Register(
            size=self.ny,
            mapping=None if output_labels is None else {k: i for i, k in enumerate(output_labels)}
            )
        self.outputs.update_from_array(
            self._func_alg(self.initial_value, np.zeros(self.nu), 0.0)
            )


    def __len__(self):
        """Algebraic passthrough from the inputs to the outputs exists if the
        output depends on the input directly, e.g. a node potential that is
        enforced by a voltage source.
        """
        return int(self._feedthrough) if self._active else 0


    # sparse solves -------------------------------------------------------------------

    def _solve(self, d, u):
        """Solve the constant sparse system for the full state and the
        derivative of the differential states. The last solution is cached since
        the dynamics and the output are evaluated at the same point.

        Parameters
        ----------
        d : array[float]
            differential states
        u : array[float]
            block input

        Returns
        -------
        x : array[float]
            full state of the DAE
        dd : array[float]
            derivative of the differential states
        """
        d = np.atleast_1d(d)
        u = np.atleast_1d(u)

        if u.size != self.nu:
            raise ValueError(
                f"'SparseLinearDAE' expects {self.nu} inputs but got {u.size}"
                )

        if self._cache is not None \
            and np.array_equal(d, self._cache[0]) and np.array_equal(u, self._cache[1]):
            return self._cache[2]

        sol = self._lu.solve(np.concatenate([self.B @ u, d]))
        res = sol[:self.n], sol[self.n:]
        self._cache = (d.copy(), u.copy(), res)
        return res


    def _func_dyn(self, x, u, t):
        """Reduced explicit ODE of the differential states."""
        return self._solve(x, u)[1]


    def _func_alg(self, x, u, t):
        """Output of the system."""
        return self.C @ self._solve(x, u)[0] + self.D @ np.atleast_1d(u)


    def _matvec(self, v):
        """Action of the reduced jacobian 'd(dd)/dd' on a vector."""
        sol = self._lu.solve(np.concatenate([np.zeros(self.n), v]))
        return sol[self.n:]


    def _newton_solve(self, c, r):
        """Solve the Newton system '(c J - I) y = r' for the reduced jacobian 'J'
        without forming it. With 'z = J y' this is 'c z - y = r', which together
        with the definition of 'J' is the sparse system

            [[S1, P], [S2r, -c I]] [x, z] = [0, -r]

        and then 'y = c z - r'.

        Parameters
        ----------
        c : float
            scale of the jacobian (timestep times integrator coefficient)
        r : array[float]
            fixed-point residual

        Returns
        -------
        y : array[float]
            solution of the Newton system
        """
        if c == 0.0:
            return -r

        lu = self._lu_newton.get(c)
        if lu is None:
            Kc = sp.bmat(
                [[self.S1, self._P], [self._S2r, -c * sp.identity(self.nd)]],
                format="csc"
                )
            lu = _factorize_sparse(Kc)
            if lu is None:
                raise RuntimeError(
                    f"the Newton matrix of 'SparseLinearDAE' is singular for scale {c}"
                    )
            if len(self._lu_newton) >= self._NEWTON_CACHE:
                self._lu_newton.pop(next(iter(self._lu_newton)))
            self._lu_newton[c] = lu

        z = lu.solve(np.concatenate([np.zeros(self.n), -r]))[self.n:]
        return c * z - r


    def _jac_alg_x(self, x, u, t):
        """Dense jacobian of the output w.r.t. the differential states, from
        'ny' adjoint solves. Only needed for linearization, evaluated lazily.
        """
        if self._jac_x_alg is None:
            rhs = np.zeros((self.n + self.nd, self.ny))
            rhs[:self.n] = self.C.T.toarray()
            self._jac_x_alg = self._lu.solve(rhs, trans="T")[self.n:].T
        return self._jac_x_alg


    # block interface -----------------------------------------------------------------

    def to_statespace(self, t):
        """Local linear state space model of the reduced system in the
        differential states :math:`d`, with dense matrices (the reduced system
        only has `nd` states).

        Parameters
        ----------
        t : float
            evaluation time

        Returns
        -------
        A, B, C, D : np.ndarray
            local state space matrices of the block
        """
        return (
            self._jac.toarray(),
            self._jac_u_dyn.copy(),
            self._jac_alg_x(None, None, t).copy(),
            self._jac_u_alg.copy()
            )


    def update(self, t):
        """Update the output, which algebraically depends on the differential
        states and the input.

        Parameters
        ----------
        t : float
            evaluation time
        """
        x, u = self.engine.state, self.inputs.to_array()
        self.outputs.update_from_array(self.op_alg(x, u, t))


    def solve(self, t, dt):
        """Advance the implicit update equation of the solver with the reduced
        right hand side and its matrix-free Jacobian.

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
        f, J = self.op_dyn(x, u, t), self.op_dyn.jac_x(x, u, t)
        return self.engine.solve(f, J, dt)


    def step(self, t, dt):
        """Compute the timestep update with the integration engine.

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
        f = self.op_dyn(x, u, t)
        return self.engine.step(f, dt)
