########################################################################################
##
##                      EXPLICIT and IMPLICIT EULER INTEGRATORS
##                                (solvers/euler.py)
##
########################################################################################

# IMPORTS ==============================================================================

from ._solver import ExplicitSolver, ImplicitSolver


# SOLVERS ==============================================================================

class EUF(ExplicitSolver):
    """Explicit forward Euler method. First-order, single-stage.

    .. math::

        x_{n+1} = x_n + h \\, f(x_n, t_n)

    Characteristics
    ---------------
    * Order: 1
    * Stages: 1
    * Explicit, fixed timestep
    * Not A-stable

    Note
    ----
    The cheapest solver per step but also the least accurate. Its small stability
    region requires very small timesteps for moderately dynamic block diagrams,
    which usually makes higher-order methods more efficient overall. Prefer
    ``RK4`` for fixed-step or ``RKDP54`` for adaptive integration of non-stiff
    systems. Only practical when computational cost per step must be absolute
    minimum and accuracy is secondary.

    References
    ----------
    .. [1] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). "Solving Ordinary
           Differential Equations I: Nonstiff Problems". Springer Series in
           Computational Mathematics, Vol. 8.
           :doi:`10.1007/978-3-540-78862-1`
    .. [2] Butcher, J. C. (2016). "Numerical Methods for Ordinary Differential
           Equations". John Wiley & Sons, 3rd Edition.
           :doi:`10.1002/9781119121534`

    """

    def step(self, f, dt):
        """performs the explicit forward timestep for (t+dt) 
        based on the state and input at (t)

        Parameters
        ----------
        f : array_like
            evaluation of function
        dt : float 
            integration timestep

        Returns
        -------
        success : bool
            timestep was successful
        err : float
            truncation error estimate
        scale : float
            timestep rescale from error controller
        """

        #get current state from history
        x_0 = self.history[0]

        #update state with euler step
        self.x = x_0 + dt * f

        #no error estimate available
        return True, 0.0, None


class EUB(ImplicitSolver):
    """Implicit backward Euler method. First-order, A-stable and L-stable.

    .. math::

        x_{n+1} = x_n + h \\, f(x_{n+1}, t_{n+1})

    The implicit equation is solved iteratively by the internal optimizer.

    Characteristics
    ---------------
    * Order: 1
    * Stages: 1 (implicit)
    * Fixed timestep
    * A-stable, L-stable

    Note
    ----
    Maximum stability at the cost of accuracy. L-stability fully damps
    parasitic high-frequency modes, making this a safe fallback for very stiff
    block diagrams (e.g. high-gain PID loops or fast electrical dynamics coupled
    to slow mechanical plant). Because each step requires solving a nonlinear
    equation, the cost per step is higher than explicit methods. For better
    accuracy on stiff systems, use ``BDF2`` (fixed-step) or ``ESDIRK43``
    (adaptive).

    References
    ----------
    .. [1] Curtiss, C. F., & Hirschfelder, J. O. (1952). "Integration of stiff
           equations". Proceedings of the National Academy of Sciences, 38(3),
           235-243. :doi:`10.1073/pnas.38.3.235`
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def solve(self, f, J, dt):
        """Solves the implicit update equation 
        using the internal optimizer.

        Parameters
        ----------
        f : array_like
            evaluation of function
        J : array_like
            evaluation of jacobian of function
        dt : float 
            integration timestep

        Returns
        -------
        err : float
            residual error of the fixed point update equation
        """

        #get current state from history
        x_0 = self.history[0]

        #update the fixed point equation
        g = x_0 + dt * f

        #use the numerical jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, dt * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err