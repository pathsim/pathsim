########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkv65.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKV65(ExplicitRungeKutta):
    """Verner 6(5) "most robust" pair. Nine stages, 6th order with
    embedded 5th order error estimate.

    Characteristics
    ---------------
    * Order: 6 (propagating) / 5 (embedded)
    * Stages: 9
    * Explicit, adaptive timestep

    Note
    ----
    Fills the gap between 5th order pairs (``RKDP54``) and the expensive 8th
    order ``RKDP87``. The extra stages pay off when the dynamics are smooth
    and tolerances are tight (roughly :math:`10^{-8}` or below), because the
    higher order allows much larger steps. For tolerances in the
    :math:`10^{-4}`--:math:`10^{-6}` range, ``RKDP54`` is usually cheaper
    overall due to fewer stages.

    References
    ----------
    .. [1] Verner, J. H. (2010). "Numerically optimal Runge-Kutta pairs
           with interpolants". Numerical Algorithms, 53(2-3), 383-396.
           :doi:`10.1007/s11075-009-9290-3`
    .. [2] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving
           Ordinary Differential Equations I: Nonstiff Problems". Springer
           Series in Computational Mathematics, Vol. 8.
           :doi:`10.1007/978-3-540-78862-1`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 9

        #order of scheme and embedded method
        self.n = 6
        self.m = 5

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 9/50, 1/6, 1/4, 53/100, 3/5, 4/5, 1.0, 1.0]

        #extended butcher table 
        self.BT = {
            0: [             9/50],
            1: [           29/324, 25/324],
            2: [             1/16,      0,           3/16],
            3: [     79129/250000,      0, -261237/250000,      19663/15625],
            4: [  1336883/4909125,      0,   -25476/30875,    194159/185250,       8225/78546],
            5: [-2459386/14727375,      0,    19504/30875, 2377474/13615875, -6157250/5773131,   902/735],
            6: [        2699/7410,      0,      -252/1235, -1393253/3993990,     236875/72618,   -135/49,   15/22], 
            7: [           11/144,      0,              0,          256/693,                0,   125/504, 125/528,        5/72], 
            8: [           11/144,      0,              0,          256/693,                0,   125/504, 125/528,        5/72]
            }
                  
        #compute coefficients for truncation error
        _A1 = [11/144, 0, 0, 256/693,              0,   125/504, 125/528,        5/72, 0]
        _A2 = [28/477, 0, 0, 212/441, -312500/366177, 2125/1764,       0, -2105/35532, 2995/17766]        
        self.TR = [a-b for a, b in zip(_A1, _A2)]