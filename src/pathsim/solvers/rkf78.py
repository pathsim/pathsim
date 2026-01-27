########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf78.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKF78(ExplicitRungeKutta):
    """Runge-Kutta-Fehlberg 7(8) pair. Thirteen stages, 7th order propagation
    with 8th order error estimate.

    Characteristics
    ---------------
    * Order: 7 (propagating) / 8 (error estimate)
    * Stages: 13
    * Explicit, adaptive timestep

    Note
    ----
    One of the earliest very-high-order embedded pairs. At the same stage
    count, the Dormand-Prince pair (``RKDP87``) generally provides better
    error constants. Consider ``RKDP87`` for new work unless Fehlberg-pair
    compatibility is required.

    References
    ----------
    .. [1] Fehlberg, E. (1968). "Classical fifth-, sixth-, seventh-, and
           eighth-order Runge-Kutta formulas with stepsize control". NASA
           Technical Report TR R-287.
    .. [2] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving
           Ordinary Differential Equations I: Nonstiff Problems". Springer
           Series in Computational Mathematics, Vol. 8.
           :doi:`10.1007/978-3-540-78862-1`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 13

        #order of scheme and embedded method
        self.n = 7
        self.m = 8

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1]

        #extended butcher table 
        self.BT = {
            0:  [      2/27],
            1:  [      1/36, 1/12],
            2:  [      1/24,    0,    1/8],
            3:  [      5/12,    0, -25/16,    25/16],
            4:  [      1/20,    0,      0,      1/4,       1/5],
            5:  [   -25/108,    0,      0,  125/108,    -65/27,  125/54],
            6:  [    31/300,    0,      0,        0,    61/225,    -2/9,    13/900],
            7:  [         2,    0,      0,    -53/6,    704/45,  -107/9,     67/90,     3],
            8:  [   -91/108,    0,      0,   23/108,  -976/135,  311/54,    -19/60,  17/6,  -1/12],
            9:  [ 2383/4100,    0,      0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41],
            10: [     3/205,    0,      0,        0,         0,   -6/41,    -3/205, -3/41,   3/41,  6/41],
            11: [-1777/4100,    0,      0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41,   0, 1],
            12: [    41/840,    0,      0,        0,         0,  34/105,      9/35,  9/35,  9/280, 9/280, 41/840]
            }

        #coefficients for local truncation error estimate
        self.TR = [41/840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41/840, -41/840, -41/840]