########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkbs32.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKBS32(ExplicitRungeKutta):
    """Bogacki-Shampine 3(2) pair. Four-stage, 3rd order with FSAL property.

    The underlying method of MATLAB's ``ode23``. The First-Same-As-Last
    (FSAL) property makes the effective cost three stages per accepted step.

    Characteristics
    ---------------
    * Order: 3 (propagating) / 2 (embedded)
    * Stages: 4 (3 effective with FSAL)
    * Explicit, adaptive timestep

    Note
    ----
    A good default when moderate accuracy suffices and per-step cost matters
    more than large step sizes. Fewer stages than 5th order pairs, so faster
    per step but needs more steps for the same global error. In a PathSim
    block diagram with smooth, non-stiff dynamics and relaxed tolerances this
    is often the most efficient explicit choice. Switch to ``RKDP54`` when
    tighter tolerances are required.

    References
    ----------
    .. [1] Bogacki, P., & Shampine, L. F. (1989). "A 3(2) pair of
           Runge-Kutta formulas". Applied Mathematics Letters, 2(4),
           321-325. :doi:`10.1016/0893-9659(89)90079-7`
    .. [2] Shampine, L. F., & Reichelt, M. W. (1997). "The MATLAB ODE
           Suite". SIAM Journal on Scientific Computing, 18(1), 1-22.
           :doi:`10.1137/S1064827594276424`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme and embedded method
        self.n = 3
        self.m = 2

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 3/4, 1.0]
        
        #extended butcher table
        self.BT = {
            0: [1/2],
            1: [0.0 , 3/4],
            2: [2/9 , 1/3, 4/9],
            3: [2/9 , 1/3, 4/9]
            }

        #coefficients for truncation error estimate
        self.TR = [-5/72, 1/12, 1/9, -1/8]