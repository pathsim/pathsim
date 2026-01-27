########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk33.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK33(ExplicitRungeKutta):
    """Three-stage, 3rd order optimal SSP Runge-Kutta method.

    The unique optimal three-stage, 3rd order SSP scheme. Commonly paired
    with WENO and ENO spatial discretisations for hyperbolic conservation
    laws.

    Characteristics
    ---------------
    * Order: 3
    * Stages: 3
    * Explicit, fixed timestep
    * SSP coefficient :math:`\\mathcal{C} = 1`

    Note
    ----
    The standard SSP time integrator for method-of-lines PDE discretisations
    inside ``ODE`` blocks. If the spatial operator is TVD under forward Euler,
    this method preserves that property at the same timestep restriction.
    When stability is borderline, ``SSPRK34`` allows roughly twice the
    timestep at the cost of one extra stage.

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
    .. [3] Gottlieb, S., Ketcheson, D. I., & Shu, C.-W. (2011). "Strong
           Stability Preserving Runge-Kutta and Multistep Time
           Discretizations". World Scientific. :doi:`10.1142/7498`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 3

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [0.0, 1.0, 0.5]

        #butcher table
        self.BT = {
            0: [1.0],
            1: [1/4, 1/4],
            2: [1/6, 1/6, 2/3]
            }

    def interpolate(self, r, dt):
        k1, k2, k3 = self.K[0], self.K[1], self.K[2]
        b1, b2, b3 = r*(2-r)**2/2, r**2*(3-2*r)/2, r**3
        return self.x_0 + dt*(b1 * k1 + b2 * k2 + b3 * k3)