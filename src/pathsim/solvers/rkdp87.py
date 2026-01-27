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

class RKDP87(ExplicitRungeKutta):
    """Dormand-Prince 8(7) pair (DOP853). Thirteen stages, 8th order with
    embedded 7th order error estimate.

    The highest-order general-purpose explicit pair in this library. Has the
    FSAL property (not exploited in this implementation).

    Characteristics
    ---------------
    * Order: 8 (propagating) / 7 (embedded)
    * Stages: 13
    * Explicit, adaptive timestep

    Note
    ----
    Only worthwhile when the dynamics are very smooth and tolerances are
    extremely tight (roughly :math:`10^{-10}` or below). The 13 function
    evaluations per step are expensive, but the 8th order convergence means
    the step size can be much larger than with lower-order methods at the
    same error. Suitable for generating reference solutions to validate other
    solvers in a block diagram. For typical engineering tolerances
    (:math:`10^{-4}`--:math:`10^{-8}`), ``RKDP54`` or ``RKV65`` are more
    efficient.

    References
    ----------
    .. [1] Prince, P. J., & Dormand, J. R. (1981). "High order embedded
           Runge-Kutta formulae". Journal of Computational and Applied
           Mathematics, 7(1), 67-75.
           :doi:`10.1016/0771-050X(81)90010-3`
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
        self.n = 8
        self.m = 7

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248/9719169821, 13/20, 1201146811/1299019798, 1.0, 1.0]

        #extended butcher table 
        self.BT = {
            0:  [1/18],
            1:  [1/48, 1/16],
            2:  [1/32, 0, 3/32],
            3:  [5/16, 0, -75/64, 75/64],
            4:  [3/80, 0, 0, 3/16, 3/20],
            5:  [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000],
            6:  [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555],
            7:  [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287],
            8:  [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789],
            9:  [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653],
            10: [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083],
            11: [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/925320556, -13158990841/6184727034, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0],
            12: [14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]
            }

        #coefficients for lower order solution evaluation
        bh = [13451932/455176623, 0, 0, 0, 0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535, 53011238/667516719, 2/45, 0]

        #coefficients for truncation error
        self.TR = [a-b for a, b in zip(self.BT[12], bh)]