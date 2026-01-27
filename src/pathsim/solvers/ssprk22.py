########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk22.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK22(ExplicitRungeKutta):
    """Two-stage, 2nd order Strong Stability Preserving (SSP) Runge-Kutta method.

    Also known as Heun's method. SSP methods preserve monotonicity and total
    variation diminishing (TVD) properties of the spatial discretisation under
    a timestep restriction scaled by the SSP coefficient.

    Characteristics
    ---------------
    * Order: 2
    * Stages: 2
    * Explicit, fixed timestep
    * SSP coefficient :math:`\\mathcal{C} = 1`

    Note
    ----
    Relevant when a block diagram wraps a method-of-lines discretisation of a
    hyperbolic PDE (e.g. shallow water, compressible Euler) inside an ``ODE``
    block and the spatial operator is TVD under forward Euler. For typical
    ODE-based block diagrams without such structure, ``RK4`` or ``RKDP54``
    are more appropriate choices.

    References
    ----------
    .. [1] Shu, C.-W., & Osher, S. (1988). "Efficient implementation of
           essentially non-oscillatory shock-capturing schemes". Journal of
           Computational Physics, 77(2), 439-471.
           :doi:`10.1016/0021-9991(88)90177-5`
    .. [2] Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). "Strong
           stability-preserving high-order time discretization methods".
           SIAM Review, 43(1), 89-112.
           :doi:`10.1137/S003614450036757X`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 2

        #order of scheme
        self.n = 2

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0]

        #butcher table
        self.BT = {
            0: [1.0],
            1: [1/2, 1/2]
            }


    def interpolate(self, r, dt):
        k1, k2 = self.K[0], self.K[1]
        b1, b2 = r*(2-r)/2, r**2/2
        return self.x_0 + dt*(b1 * k1 + b2 * k2)