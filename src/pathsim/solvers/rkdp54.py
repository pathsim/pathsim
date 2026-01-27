########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                (solvers/rkdp54.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKDP54(ExplicitRungeKutta):
    """Dormand-Prince 5(4) pair (DOPRI5). Seven stages, 5th order with
    embedded 4th order error estimate.

    The industry-standard adaptive explicit solver and the basis of MATLAB's
    ``ode45``. Has the FSAL property (not exploited in this implementation,
    so all seven stages are evaluated each step).

    Characteristics
    ---------------
    * Order: 5 (propagating) / 4 (embedded)
    * Stages: 7
    * Explicit, adaptive timestep

    Note
    ----
    Recommended default for non-stiff block diagrams. Handles smooth
    nonlinear dynamics, coupled oscillators, and signal-processing chains
    efficiently. If the simulation warns about excessive step rejections or
    very small timesteps, the system is likely stiff and an implicit solver
    (``ESDIRK43``, ``GEAR52A``) should be used instead. For very tight
    tolerances on smooth problems, ``RKV65`` or ``RKDP87`` can be more
    efficient per unit accuracy.

    References
    ----------
    .. [1] Dormand, J. R., & Prince, P. J. (1980). "A family of embedded
           Runge-Kutta formulae". Journal of Computational and Applied
           Mathematics, 6(1), 19-26.
           :doi:`10.1016/0771-050X(80)90013-3`
    .. [2] Shampine, L. F., & Reichelt, M. W. (1997). "The MATLAB ODE
           Suite". SIAM Journal on Scientific Computing, 18(1), 1-22.
           :doi:`10.1137/S1064827594276424`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 7

        #order of scheme and embedded method
        self.n = 5
        self.m = 4

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
        
        #extended butcher table
        self.BT = {
            0: [       1/5],
            1: [      3/40,        9/40],
            2: [     44/45,      -56/15,       32/9], 
            3: [19372/6561, -25360/2187, 64448/6561, -212/729],
            4: [ 9017/3168,     -355/33, 46732/5247,   49/176, -5103/18656],
            5: [    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84],
            6: [    35/384,           0,   500/1113,  125/192,  -2187/6784, 11/84]
            }

        #coefficients for local truncation error estimate
        self.TR = [71/57600, 0, - 71/16695, 71/1920, -17253/339200, 22/525, -1/40]