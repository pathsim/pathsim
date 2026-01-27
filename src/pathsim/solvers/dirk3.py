########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/dirk3.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class DIRK3(DiagonallyImplicitRungeKutta):
    """Four-stage, 3rd order L-stable DIRK method. Stiffly accurate.

    L-stability (:math:`|R(\\infty)| = 0`) fully damps parasitic
    high-frequency modes. The stiffly accurate property ensures the last
    stage equals the step output, which is beneficial for
    differential-algebraic systems.

    Characteristics
    ---------------
    * Order: 3
    * Stages: 4 (implicit)
    * Fixed timestep
    * L-stable, stiffly accurate

    Note
    ----
    A robust fixed-step solver for stiff block diagrams. L-stability makes
    it well-suited for systems with widely separated time scales, such as a
    fast electrical subsystem driving a slow thermal or mechanical model.
    Also used internally as the startup method for ``BDF`` solvers. For
    adaptive stepping on stiff problems, prefer ``ESDIRK43``.

    References
    ----------
    .. [1] Alexander, R. (1977). "Diagonally implicit Runge-Kutta methods
           for stiff O.D.E.'s". SIAM Journal on Numerical Analysis, 14(6),
           1006-1021. :doi:`10.1137/0714068`
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [1/2, 2/3, 1/2, 1.0]

        #butcher table
        self.BT = {
            0: [1/2],
            1: [1/6, 1/2], 
            2: [-1/2, 1/2, 1/2], 
            3: [3/2, -3/2, 1/2, 1/2]
            }