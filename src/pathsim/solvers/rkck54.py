########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkck54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKCK54(ExplicitRungeKutta):
    """Cash-Karp 5(4) pair. Six stages, 5th order with embedded 4th order
    error estimate.

    Designed to improve on the stability properties of the Fehlberg pair
    (``RKF45``) while keeping the same stage count.

    Characteristics
    ---------------
    * Order: 5 (propagating) / 4 (embedded)
    * Stages: 6
    * Explicit, adaptive timestep

    Note
    ----
    Comparable to ``RKDP54`` in cost and accuracy for most non-stiff block
    diagrams. Can exhibit slightly better stability on problems with
    eigenvalues near the imaginary axis. Both pairs are solid 5th order
    choices; ``RKDP54`` is the more commonly used default.

    References
    ----------
    .. [1] Cash, J. R., & Karp, A. H. (1990). "A variable order Runge-Kutta
           method for initial value problems with rapidly varying right-hand
           sides". ACM Transactions on Mathematical Software, 16(3), 201-222.
           :doi:`10.1145/79505.79507`
    .. [2] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving
           Ordinary Differential Equations I: Nonstiff Problems". Springer
           Series in Computational Mathematics, Vol. 8.
           :doi:`10.1007/978-3-540-78862-1`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme and embedded method
        self.n = 5
        self.m = 4

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/5, 3/10, 3/5, 1, 7/8]

        #extended butcher table 
        self.BT = {
            0: [       1/5],
            1: [      3/40,    9/40],
            2: [      3/10,   -9/10,       6/5],
            3: [    -11/54,     5/2,    -70/27,        35/27],
            4: [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096],
            5: [    37/378,       0,   250/621,      125/594,        0, 512/1771]
            }

        #coefficients for local truncation error estimate
        self.TR = [-277/64512, 0, 6925/370944, -6925/202752, -277/14336, 277/7084]