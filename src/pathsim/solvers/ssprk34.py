########################################################################################
##
##               EXPLICIT STRONG STABILITY PRESERVING RUNGE-KUTTA INTEGRATOR
##                                (solvers/ssprk34.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class SSPRK34(ExplicitRungeKutta):
    """Four-stage, 3rd order SSP Runge-Kutta method with SSP coefficient 2.

    An extra stage compared to ``SSPRK33`` doubles the allowable SSP timestep
    (:math:`\\mathcal{C} = 2`), giving a larger effective stability region
    along the negative real axis.

    Characteristics
    ---------------
    * Order: 3
    * Stages: 4
    * Explicit, fixed timestep
    * SSP coefficient :math:`\\mathcal{C} = 2`

    Note
    ----
    Preferable over ``SSPRK33`` when a method-of-lines ``ODE`` block is close
    to the SSP timestep limit and the cost of one additional stage per step is
    acceptable in exchange for a factor-of-two relaxation in the CFL
    constraint.

    References
    ----------
    .. [1] Spiteri, R. J., & Ruuth, S. J. (2002). "A new class of optimal
           high-order strong-stability-preserving time discretization methods".
           SIAM Journal on Numerical Analysis, 40(2), 469-491.
           :doi:`10.1137/S0036142901389025`
    .. [2] Gottlieb, S., Ketcheson, D. I., & Shu, C.-W. (2011). "Strong
           Stability Preserving Runge-Kutta and Multistep Time
           Discretizations". World Scientific. :doi:`10.1142/7498`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 3

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/2, 1, 1/2]

        #butcher table
        self.BT = {
            0: [1/2],
            1: [1/2, 1/2],
            2: [1/6, 1/6, 1/6],
            3: [1/6, 1/6, 1/6, 1/2]
            }