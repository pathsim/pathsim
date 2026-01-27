########################################################################################
##
##                       DIAGONALLY IMPLICIT RUNGE KUTTA METHOD
##                                (solvers/esdirk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import DiagonallyImplicitRungeKutta


# SOLVERS ==============================================================================

class ESDIRK4(DiagonallyImplicitRungeKutta):
    """Six-stage, 4th order ESDIRK method. L-stable and stiffly accurate.

    No embedded error estimator; fixed timestep only.

    Characteristics
    ---------------
    * Order: 4
    * Stages: 6 (1 explicit, 5 implicit)
    * Fixed timestep
    * L-stable, stiffly accurate
    * Stage order 2

    Note
    ----
    Provides 4th order accuracy on stiff block diagrams when the timestep is
    predetermined (e.g. real-time or hardware-in-the-loop contexts). The
    explicit first stage reuses the last function evaluation from the
    previous step, saving one implicit solve per step compared to a fully
    implicit DIRK. L-stability and stiff accuracy ensure full damping of
    parasitic high-frequency modes. For adaptive stepping, use ``ESDIRK43``
    which adds an embedded error estimator at the same stage count.

    References
    ----------
    .. [1] Kennedy, C. A., & Carpenter, M. H. (2016). "Diagonally implicit
           Runge-Kutta methods for ordinary differential equations. A
           review". NASA/TM-2016-219173.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme
        self.n = 4

        #intermediate evaluation times
        self.eval_stages = [
            0.0, 1/2, 1/6, 37/40, 1/2, 1.0
            ]

        #butcher table
        self.BT = {
            0: None, #explicit first stage
            1: [1/4, 1/4],
            2: [-1/36, -1/18, 1/4],
            3: [-21283/32000, -5143/64000, 90909/64000, 1/4],
            4: [46010759/749250000, -737693/40500000, 10931269/45500000, -1140071/34090875, 1/4],
            5: [89/444, 89/804756, -27/364, -20000/171717, 843750/1140071, 1/4]
            }