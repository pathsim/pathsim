########################################################################################
##
##                       CLASSICAL EXPLICIT RUNGE-KUTTA INTEGRATOR
##                                 (solvers/rk4.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RK4(ExplicitRungeKutta):
    """Classical four-stage, 4th order explicit Runge-Kutta method.

    .. math::

        k_1 &= f(x_n,\\; t_n) \\\\
        k_2 &= f\\!\\left(x_n + \\tfrac{h}{2}\\,k_1,\\; t_n + \\tfrac{h}{2}\\right) \\\\
        k_3 &= f\\!\\left(x_n + \\tfrac{h}{2}\\,k_2,\\; t_n + \\tfrac{h}{2}\\right) \\\\
        k_4 &= f(x_n + h\\,k_3,\\; t_n + h) \\\\
        x_{n+1} &= x_n + \\tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Characteristics
    ---------------
    * Order: 4
    * Stages: 4
    * Explicit, fixed timestep

    Note
    ----
    The standard fixed-step explicit solver. Provides a good cost-to-accuracy
    ratio for non-stiff block diagrams where the timestep is known a priori
    (e.g. real-time or hardware-in-the-loop simulation with a fixed clock).
    Not suitable for stiff systems. When accuracy demands vary during a run,
    adaptive methods like ``RKDP54`` are more efficient because they
    concentrate steps where the dynamics change rapidly.

    References
    ----------
    .. [1] Kutta, W. (1901). "Beitrag zur näherungsweisen Integration totaler
           Differentialgleichungen". Zeitschrift für Mathematik und Physik,
           46, 435-453.
    .. [2] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in
           Computational Mathematics, Vol. 8.
           :doi:`10.1007/978-3-540-78862-1`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 4

        #order of scheme
        self.n = 4

        #intermediate evaluation times
        self.eval_stages = [0.0, 0.5, 0.5, 1.0]

        #butcher table
        self.BT = {
            0: [1/2],
            1: [0.0, 1/2],
            2: [0.0, 0.0, 1.0], 
            3: [1/6, 2/6, 2/6, 1/6]
            }


    def interpolate(self, r, dt):
        k1, k2, k3, k4 = self.K[0], self.K[1], self.K[2], self.K[3]
        b1, b2, b3, b4 = r*(1-r)**2/6, r**2*(2-3*r)/2, r**2*(3*r-4)/2, r**3/6
        return self.x_0 + dt*(b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4)