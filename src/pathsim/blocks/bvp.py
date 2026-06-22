#########################################################################################
##
##                           BOUNDARY VALUE PROBLEM BLOCK
##                                (pathsim/blocks/bvp.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from scipy.integrate import solve_bvp

from ._block import Block


# BLOCKS ================================================================================

class BVP1D(Block):
    """One-dimensional two-point boundary value problem (BVP) block.

    Solves a first-order system of ordinary differential equations on a spatial
    domain :math:`[a, b]` subject to two-point boundary conditions

    .. math::

        \\begin{align}
            y'(x) &= \\mathrm{fun}(x, y, p, u) \\\\
                0 &= \\mathrm{bc}(y(a), y(b), p, u)
        \\end{align}


    with optional free parameters :math:`p` that are determined together with
    the solution (e.g. eigenvalue problems). Here :math:`u` is the block input,
    so the boundary data and right hand side can depend on external signals, and
    the prime denotes the derivative with respect to the spatial coordinate
    :math:`x`.

    The problem is solved with :func:`scipy.integrate.solve_bvp` (a 4th-order
    collocation method with residual based mesh refinement). At every evaluation
    the solver is warm-started with the mesh, solution and parameters of the
    previous solve, so slowly varying inputs are tracked cheaply. When the input
    is unchanged since the last successful solve the re-solve is skipped
    entirely, so repeated evaluations within a timestep (e.g. during the
    algebraic loop) do not recompute the solution. The block output is the
    solution sampled at the query points `x_eval` (row-major, equation by
    equation) followed by the converged free parameters.

    Note
    ----
    Interior / multipoint conditions are not supported by
    :func:`scipy.integrate.solve_bvp` and are therefore not available in this
    block. If a solve does not converge, the previous output and warm-start are
    retained and the `success` attribute is set to `False`.

    Example
    -------
    Solve :math:`y'' = -y` with :math:`y(0) = 0`, :math:`y(\\pi/2) = 1`, whose
    solution is :math:`y(x) = \\sin(x)`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import BVP1D

        #first-order system y0' = y1, y1' = -y0
        def fun(x, y, p, u):
            return np.vstack([y[1], -y[0]])

        def bc(ya, yb, p, u):
            return np.array([ya[0], yb[0] - 1.0])

        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2))


    An eigenvalue problem :math:`y'' + p^2 y = 0`, :math:`y(0) = y(1) = 0`,
    :math:`y'(0) = 1`, with the free parameter `p` converging to :math:`\\pi`:

    .. code-block:: python

        import numpy as np
        from pathsim.blocks import BVP1D

        def fun(x, y, p, u):
            return np.vstack([y[1], -p[0]**2 * y[0]])

        def bc(ya, yb, p, u):
            return np.array([ya[0], yb[0], ya[1] - 1.0])

        bvp = BVP1D(fun, bc, n=2, domain=(0.0, 1.0), p0=[3.0],
                    y0=lambda x: np.vstack([np.sin(np.pi*x), np.pi*np.cos(np.pi*x)]))


    Parameters
    ----------
    fun : callable
        right hand side of the spatial ODE system with signature
        `fun(x, y, p, u)` where `x` is the mesh `(m,)`, `y` is `(n, m)`, `p` are
        the free parameters and `u` is the block input, returning `(n, m)`
    bc : callable
        boundary condition residuals with signature `bc(ya, yb, p, u)` returning
        an array of dimension `n + n_p`
    n : int
        number of first-order equations
    domain : tuple[float, float]
        spatial domain `(a, b)`
    n_nodes : int
        number of nodes in the initial mesh
    x_eval : array[float], None
        query points for the output, defaults to `n_nodes` points spanning the
        domain
    y0 : array[float], callable, None
        initial guess for the solution, either an `(n, n_nodes)` array, a
        callable `y0(x)` returning `(n, m)`, or `None` for an all-zero guess
    p0 : array[float], None
        initial guess for the free parameters, `None` if there are none
    tol : float
        solver tolerance passed to :func:`scipy.integrate.solve_bvp`

    Attributes
    ----------
    success : bool
        whether the most recent solve converged
    x : array[float]
        current (refined) spatial mesh
    """

    def __init__(
        self,
        fun=lambda x, y, p, u: np.zeros_like(y),
        bc=lambda ya, yb, p, u: ya,
        n=1,
        domain=(0.0, 1.0),
        n_nodes=11,
        x_eval=None,
        y0=None,
        p0=None,
        tol=1e-6
        ):

        super().__init__()

        #spatial ode and boundary condition residuals
        self.fun = fun
        self.bc = bc

        #problem dimensions
        self.n = int(n)
        a, b = domain
        self.domain = (float(a), float(b))

        #free parameters
        self._has_p = p0 is not None
        self.p0 = None if p0 is None else np.atleast_1d(p0).astype(float)

        #query points for the output
        self.x_eval = (np.linspace(a, b, n_nodes) if x_eval is None
                       else np.asarray(x_eval, dtype=float))

        #solver tolerance
        self.tol = tol

        #initial mesh and solution guess (used as warm-start)
        self._x0 = np.linspace(a, b, n_nodes)
        if y0 is None:
            self._y0 = np.zeros((self.n, n_nodes))
        elif callable(y0):
            self._y0 = np.atleast_2d(y0(self._x0)).astype(float)
        else:
            self._y0 = np.broadcast_to(
                np.atleast_2d(y0), (self.n, n_nodes)
                ).astype(float).copy()

        #warm-start state
        self._x = self._x0.copy()
        self._y = self._y0.copy()
        self._p = None if self.p0 is None else self.p0.copy()

        #solve status and sampled solution of the most recent solve
        self.success = False
        self._y_eval = np.zeros((self.n, self.x_eval.size))

        #input of the most recent successful solve, used to skip redundant
        #re-solves when the boundary data has not changed between evaluations
        self._u_last = None

        #pre-size the output register
        n_out = self.n * self.x_eval.size + (self._p.size if self._has_p else 0)
        self.outputs.update_from_array(np.zeros(n_out))


    def __len__(self):
        #boundary data flows in through the input, so the block has a passthrough
        return 1 if self._active else 0


    def reset(self):
        """Reset inputs, outputs and the warm-start mesh, solution and
        parameters to their initial values.
        """
        super().reset()
        self._x = self._x0.copy()
        self._y = self._y0.copy()
        self._p = None if self.p0 is None else self.p0.copy()
        self.success = False
        self._u_last = None


    def solution(self):
        """Return the most recent solution sampled at the query points.

        Returns
        -------
        y : array[array[float]]
            solution of shape `(n, len(x_eval))`
        """
        return self._y_eval.copy()


    def parameters(self):
        """Return the most recent converged free parameters.

        Returns
        -------
        p : array[float], None
            converged free parameters, `None` if there are none
        """
        return None if self._p is None else self._p.copy()

    @property
    def x(self):
        return self._x


    def update(self, t):
        """Solve the boundary value problem for the current input and expose the
        sampled solution and converged parameters at the output.

        Parameters
        ----------
        t : float
            evaluation time
        """

        #current block input
        u = self.inputs.to_array()

        #skip the solve if the boundary data is unchanged since the last
        #successful solve; the warm-start and output are still valid
        if self._u_last is not None and np.array_equal(u, self._u_last):
            return

        #scipy callbacks, with or without free parameters
        if self._has_p:
            _fun = lambda x, y, p: self.fun(x, y, p, u)
            _bc = lambda ya, yb, p: self.bc(ya, yb, p, u)
            sol = solve_bvp(_fun, _bc, self._x, self._y, p=self._p, tol=self.tol)
        else:
            _fun = lambda x, y: self.fun(x, y, None, u)
            _bc = lambda ya, yb: self.bc(ya, yb, None, u)
            sol = solve_bvp(_fun, _bc, self._x, self._y, tol=self.tol)

        self.success = bool(sol.success)

        #keep the previous output and warm-start if the solve failed
        if not self.success:
            return

        #warm-start the next solve with the refined mesh and solution, and
        #remember the input so an unchanged re-evaluation can be skipped
        self._x, self._y = sol.x, sol.y
        if self._has_p:
            self._p = sol.p
        self._u_last = u.copy()

        #sample the solution at the query points and assemble the output
        self._y_eval = sol.sol(self.x_eval)[:self.n]
        y_flat = self._y_eval.ravel()
        if self._has_p:
            y_flat = np.concatenate([y_flat, self._p])
        self.outputs.update_from_array(y_flat)
