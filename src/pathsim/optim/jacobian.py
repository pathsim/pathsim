########################################################################################
##
##                          MATRIX-FREE / IMPLICIT JACOBIANS
##                                (optim/jacobian.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# CLASSES ==============================================================================

class NewtonOperator:
    """Jacobian of a block that is never formed as a matrix, but knows how to
    solve its own Newton system.

    The implicit integration engines hand the optimizer a *scaled* Jacobian
    :math:`c\\,J` (for example ``dt * J`` for implicit Euler or ``F * dt * J``
    for BDF) and the :class:`.NewtonAnderson` optimizer then solves the Newton
    system

    .. math::

        (c\\,J - I)\\,y = r

    For blocks where :math:`J` is dense but its action is cheap, for example
    when it is the Schur complement of a large sparse system, forming
    :math:`J` is wasteful or impossible. Such a block supplies

    * ``newton_solve(c, r)``: returns :math:`y = (c J - I)^{-1} r`
    * ``matvec(v)``: returns :math:`J v` (only needed for linearization)

    and passes a ``NewtonOperator`` to the engine instead of an array. The
    operator supports multiplication with a scalar from either side (this is
    all the engines do with the Jacobian), which just rescales ``c``.

    Example
    -------
    A diagonal Jacobian without ever forming it as a matrix:

    .. code-block:: python

        import numpy as np
        from pathsim.optim.jacobian import NewtonOperator

        d = np.array([-1.0, -10.0])

        J = NewtonOperator(
            newton_solve=lambda c, r: r / (c * d - 1.0),
            matvec=lambda v: d * v,
            shape=(2, 2)
            )

        #what an engine does: scale by the timestep and hand it to the optimizer
        Jh = 0.1 * J

    Parameters
    ----------
    newton_solve : callable
        ``newton_solve(c, r)`` returns :math:`(c J - I)^{-1} r`, called with the
        accumulated scalar factor ``c`` and the fixed-point residual ``r``
    matvec : callable, None
        ``matvec(v)`` returns :math:`J v`
    shape : tuple[int], None
        shape of the (square) Jacobian
    scale : float
        scalar factor ``c`` that the operator carries, ``1.0`` for the
        unscaled Jacobian
    """

    #make numpy defer to the reflected operators ('np.float64(0.1) * J')
    __array_ufunc__ = None
    __array_priority__ = 1000

    def __init__(self, newton_solve, matvec=None, shape=None, scale=1.0):
        self._newton_solve = newton_solve
        self._matvec = matvec
        self.shape = shape
        self.scale = float(scale)


    @property
    def ndim(self):
        return 2


    def _scaled(self, c):
        return NewtonOperator(
            self._newton_solve, self._matvec, self.shape, self.scale * float(c)
            )


    def __mul__(self, c):
        if np.ndim(c) != 0:
            return NotImplemented
        return self._scaled(c)


    __rmul__ = __mul__


    def __neg__(self):
        return self._scaled(-1.0)


    def newton_solve(self, r):
        """Solve the Newton system :math:`(c J - I) y = r` for the scale
        :math:`c` carried by the operator.

        Parameters
        ----------
        r : array[float]
            right hand side, the fixed-point residual

        Returns
        -------
        y : array[float]
            solution of the Newton system
        """
        return self._newton_solve(self.scale, np.asarray(r, dtype=float).ravel())


    def __matmul__(self, v):
        """Action of the (scaled) Jacobian on a vector."""
        if self._matvec is None:
            raise NotImplementedError("this NewtonOperator has no 'matvec'")
        return self.scale * np.asarray(self._matvec(np.asarray(v, dtype=float).ravel()))


    def toarray(self):
        """Dense matrix of the (scaled) Jacobian by applying it to the unit
        vectors. Only meant for debugging and testing small systems.

        Returns
        -------
        J : np.ndarray
            dense (scaled) Jacobian
        """
        if self.shape is None:
            raise ValueError("'shape' is required to build the dense matrix")
        n = self.shape[1]
        return np.column_stack([self @ e for e in np.eye(n)])
