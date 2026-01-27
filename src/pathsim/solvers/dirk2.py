########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk2.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class DIRK2(DiagonallyImplicitRungeKutta):
    """Two-stage, 2nd order DIRK method. L-stable, SSP-optimal, symplectic.

    Characteristics
    ---------------
    * Order: 2
    * Stages: 2 (implicit)
    * Fixed timestep
    * L-stable, SSP-optimal, symplectic

    Note
    ----
    The simplest multi-stage implicit Runge-Kutta method. L-stability
    fully damps parasitic high-frequency modes, and the symplectic property
    preserves Hamiltonian structure when the dynamics are conservative. Two
    implicit stages per step is relatively cheap. For higher accuracy on
    stiff systems, use ``DIRK3`` or the adaptive ``ESDIRK43``.

    References
    ----------
    .. [1] Ferracina, L., & Spijker, M. N. (2008). "Strong stability of
           singly-diagonally-implicit Runge-Kutta methods". Applied Numerical
           Mathematics, 58(11), 1675-1686.
           :doi:`10.1016/j.apnum.2007.10.004`
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 2

        #order of scheme
        self.n = 2

        #intermediate evaluation times
        self.eval_stages = [1/4, 3/4]

        #butcher table
        self.BT = {
            0: [1/4],
            1: [1/2, 1/4]
            }

        #final evaluation
        self.A = [1/2, 1/2]