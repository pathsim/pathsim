########################################################################################
##
##                            GEAR-type INTEGRATION METHODS 
##                                 (solvers/gear.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque

from ._solver import ImplicitSolver
from .esdirk32 import ESDIRK32

from .._constants import (
    TOLERANCE, 
    SOL_BETA, 
    SOL_SCALE_MIN,
    SOL_SCALE_MAX
    )


# HELPERS ==============================================================================

def compute_bdf_coefficients(order, timesteps):
    """Computes the coefficients for backward differentiation formulas for a given order.
    The timesteps can be specified for variable timestep BDF methods. 

    For m-th order BDF we have for the n-th timestep:
        sum(alpha_i * x_i; i=n-m,...,n) = h_n * f_n(x_n, t_n)
    or 
        x_n = beta * h_n * f_n(x_n, t_n) - sum(alpha_j * x_{n-1-j}; j=0,...,order-1)

    Parameters
    ----------
    order : int
        order of the integration scheme
    timesteps : array[float]
        timestep buffer (h_{n-j}; j=0,...,order-1)
    
    Returns
    ------- 
    beta : float
        weight for function
    alpha : array[float]
        weights for previous solutions
    """

    #check if valid order
    if order < 1:
        raise RuntimeError(f"BDF coefficients of order '{order}' not possible!")

    #quit early for no buffer (euler backward)
    if len(timesteps) < 2:
        return 1.0, [1.0]

    # Compute timestep ratios rho_j = h_{n-j} / h_n
    rho = timesteps[1:] / timesteps[0]

    # Compute normalized time differences theta_j
    theta = -np.ones(order + 1)
    theta[0] = 0
    for j in range(2, order + 1):
        theta[j] -= sum(rho[:j - 1])

    # Set up the linear system (p + 1 equations)
    A = np.zeros((order + 1, order + 1))
    b = np.zeros(order + 1)
    b[1] = 1 
    for m in range(order + 1):
        A[m, :] = theta ** m 

    # Solve the linear system A * alpha = b
    alphas = np.linalg.solve(A, b)

    #return function and buffer weights
    return 1 / alphas[0], -alphas[1:] / alphas[0]


# BASE GEAR SOLVER =====================================================================

class GEAR(ImplicitSolver):
    """Base class for GEAR-type integrators that defines the universal methods.

    Numerical integration method based on BDFs (linear multistep methods). 
    Uses n-th order BDF for timestepping and (n-1)-th order BDF coefficients 
    to estimate a lower ordersolutuin for error control. 

    The adaptive timestep BDF coefficients are dynamically computed at the 
    beginning of each timestep from the buffered previous timsteps.

    Notes
    -----
    Not to be used directly!

    Attributes
    ----------
    x : numeric, array[numeric]
        internal 'working' state
    n : int
        order of integration scheme
    s : int
        number of internal intermediate stages
    stage : int
        counter for current intermediate stage
    eval_stages : list[float]
        rations for evaluation times of intermediate stages
    opt : NewtonAnderson, Anderson, etc.
        optimizer instance to solve the implicit update equation
    K : dict[int: list[float]]
        bdf coefficients for the state buffer for each order
    F : dict[int: float]
        bdf coefficients for the function 'func' for each order
    history : deque[numeric]
        internal history of past results
    history_dt : deque[numeric]
        internal history of past timesteps
    startup : Solver
        internal solver instance for startup (building history) 
        of multistep methods (using 'ESDIRK32' for 'GEAR' methods)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = None
        self.m = None

        #safety factor for error controller (if available)
        self.beta = SOL_BETA

        #gear timestep buffer
        self.history_dt = deque([], maxlen=1)

        #flag adaptive timestep solver
        self.is_adaptive = True

        #initialize startup solver from 'self'
        self._needs_startup = True
        self.startup = ESDIRK32.cast(self, self.parent)


    @classmethod
    def cast(cls, other, parent, **solver_kwargs):
        """cast to this solver needs special handling of startup method

        Parameters
        ----------
        other : Solver
            solver instance to cast new instance of this class from
        parent : None | Solver
            solver instance to use as parent
        solver_kwargs : dict
            other args for the solver

        Returns
        -------
        engine : GEAR
            instance of `GEAR` solver with params and state from `other`
        """
        engine = super().cast(other, parent, **solver_kwargs)
        engine.startup = ESDIRK32.cast(engine, parent)

        return engine


    @classmethod
    def create(cls, initial_value, parent=None, from_engine=None, **solver_kwargs):
        """Create a new GEAR solver, properly initializing the startup solver.

        Parameters
        ----------
        initial_value : float, array
            initial condition / integration constant
        parent : None | Solver
            parent solver instance for stage synchronization
        from_engine : None | Solver
            existing solver to inherit state and settings from
        solver_kwargs : dict
            additional args for the solver

        Returns
        -------
        engine : GEAR
            new GEAR solver instance
        """
        if from_engine is not None:
            #inherit tolerances from existing engine if not specified
            if "tolerance_lte_rel" not in solver_kwargs:
                solver_kwargs["tolerance_lte_rel"] = from_engine.tolerance_lte_rel
            if "tolerance_lte_abs" not in solver_kwargs:
                solver_kwargs["tolerance_lte_abs"] = from_engine.tolerance_lte_abs

            #create new solver (this initializes startup in __init__)
            engine = cls(initial_value, parent, **solver_kwargs)

            #preserve state from old engine
            engine.state = from_engine.state

            #re-initialize startup solver from the new engine
            engine.startup = ESDIRK32.create(initial_value, parent, **solver_kwargs)
            engine.startup.state = from_engine.state

            return engine

        #simple creation without existing engine
        return cls(initial_value, parent, **solver_kwargs)


    def stages(self, t, dt):
        """Generator that yields the intermediate evaluation 
        time during the timestep 't + ratio * dt'.

        Parameters
        ----------
        t : float 
            evaluation time
        dt : float
            integration timestep
        """

        #not enough history for full order -> stages of startup method
        if self._needs_startup:
            for self.stage, _t in enumerate(self.startup.stages(t, dt)):
                yield _t
        else:
            for _t in super().stages(t, dt):
                yield _t


    def reset(self, initial_value=None):
        """"Resets integration engine to initial value, 
        optionally provides new initial value
    
        Parameters
        ----------
        initial_value : None | float | np.ndarray
            new initial value of the engine, optional
        """

        #update initial value if provided
        if initial_value is not None:
            self.initial_value = initial_value

        #clear buffers
        self.history.clear()
        self.history_dt.clear()

        #overwrite state with initial value (ensure array format)
        self.x = np.atleast_1d(self.initial_value).copy()

        #reset startup solver
        self.startup.reset(initial_value)


    def buffer(self, dt):
        """Buffer the state and timestep. Dynamically precompute 
        the variable timestep BDF coefficients on the fly for the 
        current timestep.
        
        Parameters
        ----------
        dt : float
            integration timestep
        """

        #reset optimizer
        self.opt.reset()
    
        #add to histories (solution and timestep)            
        self.history.appendleft(self.x)
        self.history_dt.appendleft(dt)

        #flag for startup method
        self._needs_startup = len(self.history) < self.n

        #buffer with startup method
        if self._needs_startup:
            self.startup.buffer(dt)

        #precompute coefficients here, where buffers are available
        self.F, self.K = {}, {}
        for n, _ in enumerate(self.history_dt, 1):
            self.F[n], self.K[n] = compute_bdf_coefficients(n, np.array(self.history_dt))


    # methods for adaptive timestep solvers --------------------------------------------

    def revert(self):
        """Revert integration engine to previous timestep, this is only 
        relevant for adaptive methods where the simulation timestep 'dt' 
        is rescaled and the engine step is recomputed with the smaller 
        timestep.
        """
        
        #reset internal state to previous state from history
        self.x = self.history.popleft() 

        #also remove latest timestep from timestep history
        _ = self.history_dt.popleft()

        #revert startup method
        if self._needs_startup:
            self.startup.revert()


    def error_controller(self, tr):
        """Compute scaling factor for adaptive timestep based on absolute and 
        relative tolerances for local truncation error. 

        Checks if the error tolerance is achieved and returns a success metric.
        
        Parameters
        ----------
        tr : array[float]
            truncation error estimate 

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error = np.abs(tr) / scale

        #compute the error norm and clip it
        error_norm = np.clip(float(np.max(scaled_error)), TOLERANCE, None)

        #determine if the error is acceptable
        success = error_norm <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm ** (1/self.n)

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        return success, error_norm, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def solve(self, f, J, dt):
        """Solves the implicit update equation using the optimizer of the engine.
        
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

        #not enough history for full order -> solve with startup method
        if self._needs_startup:
            err = self.startup.solve(f, J, dt)
            self.x = self.startup.get()
            return err
        
        #fixed-point function update (faster then sum comprehension)
        g = self.F[self.n] * dt * f
        for b, k in zip(self.history, self.K[self.n]):
            g = g + b * k

        #use the jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, self.F[self.n] * dt * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, f, dt):
        """Finalizes the timestep by resetting the solver for the implicit 
        update equation and computing the lower order estimate of the 
        solution for error control.

        Parameters
        ----------
        f : array_like
            evaluation of function
        dt : float 
            integration timestep

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #not enough history for full order -> step with startup method
        if self._needs_startup:
            suc, err, scl = self.startup.step(f, dt)
            self.x = self.startup.get()
            return suc, err, scl

        #estimate truncation error from lower order solution
        tr = self.x - self.F[self.m] * dt * f
        for b, k in zip(self.history, self.K[self.m]):
            tr = tr - b * k

        #error control
        return self.error_controller(tr)


# SOLVERS ==============================================================================

class GEAR21(GEAR):
    """Variable-step 2nd order BDF with 1st order error estimate. A-stable.

    BDF coefficients are recomputed each step to account for variable
    timesteps. Uses ``ESDIRK32`` as startup solver.

    Characteristics
    ---------------
    * Order: 2 (stepping) / 1 (error estimate)
    * Implicit variable-step multistep
    * Adaptive timestep
    * A-stable

    Note
    ----
    The simplest adaptive multistep stiff solver. A-stability makes it safe
    for any stiff block diagram. The multistep approach reuses past solution
    values, so per-step cost is lower than single-step implicit methods
    (ESDIRK), but a startup phase is needed to fill the history buffer. For
    higher accuracy, use ``GEAR32`` or ``ESDIRK43``.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 2
        self.m = 1

        #gear buffers, here 2
        self.history = deque([], maxlen=2)
        self.history_dt = deque([], maxlen=2)


class GEAR32(GEAR):
    """Variable-step 3rd order BDF with 2nd order error estimate.
    :math:`A(\\alpha)`-stable.

    Uses ``ESDIRK32`` as startup solver.

    Characteristics
    ---------------
    * Order: 3 (stepping) / 2 (error estimate)
    * Implicit variable-step multistep
    * Adaptive timestep
    * :math:`A(\\alpha)`-stable (BDF3 stability wedge)

    Note
    ----
    Good balance of accuracy and stability for stiff block diagrams. The
    stability wedge is nearly as wide as ``GEAR21`` (:math:`\\approx 86°`)
    while providing an extra order of accuracy. For most stiff systems this
    is a practical default when a multistep solver is preferred over ESDIRK.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 3
        self.m = 2

        #gear buffers, here 3
        self.history = deque([], maxlen=3)
        self.history_dt = deque([], maxlen=3)


class GEAR43(GEAR):
    """Variable-step 4th order BDF with 3rd order error estimate.
    :math:`A(\\alpha)`-stable.

    Uses ``ESDIRK32`` as startup solver.

    Characteristics
    ---------------
    * Order: 4 (stepping) / 3 (error estimate)
    * Implicit variable-step multistep
    * Adaptive timestep
    * :math:`A(\\alpha)`-stable (BDF4 stability wedge, :math:`\\approx 73°`)

    Note
    ----
    Narrower stability wedge than ``GEAR32``. Eigenvalues near the imaginary
    axis may be poorly damped. Use only when the stiff modes are strongly
    dissipative and 4th order accuracy is needed. Otherwise, ``GEAR32`` or
    ``ESDIRK43`` are safer choices.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 4
        self.m = 3

        #gear buffers, here 4
        self.history = deque([], maxlen=4)
        self.history_dt = deque([], maxlen=4)


class GEAR54(GEAR):
    """Variable-step 5th order BDF with 4th order error estimate.
    :math:`A(\\alpha)`-stable.

    Uses ``ESDIRK32`` as startup solver.

    Characteristics
    ---------------
    * Order: 5 (stepping) / 4 (error estimate)
    * Implicit variable-step multistep
    * Adaptive timestep
    * :math:`A(\\alpha)`-stable (BDF5 stability wedge, :math:`\\approx 51°`)

    Note
    ----
    The stability wedge is significantly narrower than lower-order GEAR
    variants. Only justified for mildly stiff problems where 5th order
    accuracy yields a clear efficiency gain. For strongly stiff systems,
    ``GEAR21``/``GEAR32`` or ``ESDIRK54`` are more robust.

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order and order of secondary method
        self.n = 5
        self.m = 4

        #gear, here 5+1
        self.history = deque([], maxlen=5)
        self.history_dt = deque([], maxlen=5)


class GEAR52A(GEAR):
    """Variable-step, variable-order BDF (orders 2--5). Adapts both timestep
    and order automatically.

    At each step the error controller compares estimates from orders
    :math:`n-1` and :math:`n+1` and selects the order that minimises the
    normalised error, allowing larger steps. Analogous to MATLAB's
    ``ode15s``. Uses ``ESDIRK32`` as startup solver.

    Characteristics
    ---------------
    * Order: variable, 2--5
    * Implicit variable-step, variable-order multistep
    * Adaptive timestep and order
    * Stability: A-stable at order 2, :math:`A(\\alpha)`-stable at orders 3--5

    Note
    ----
    The most autonomous stiff solver in this library. Automatically selects
    higher orders in smooth regions for larger steps and drops to low order
    in stiff or transient regions for stability. A good default when the
    character of the block diagram is unknown or changes during the
    simulation (e.g. switching events, varying loads).

    References
    ----------
    .. [1] Gear, C. W. (1971). "Numerical Initial Value Problems in Ordinary
           Differential Equations". Prentice-Hall.
    .. [2] Shampine, L. F., & Reichelt, M. W. (1997). "The MATLAB ODE
           Suite". SIAM Journal on Scientific Computing, 18(1), 1-22.
           :doi:`10.1137/S1064827594276424`
    .. [3] Hairer, E., & Wanner, G. (1996). "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems". Springer
           Series in Computational Mathematics, Vol. 14.
           :doi:`10.1007/978-3-642-05221-7`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #initial integration order
        self.n = 2

        #minimum and maximum BDF order to select
        self.n_min, self.n_max = 2, 5

        #gear, here 6
        self.history = deque([], maxlen=6)
        self.history_dt = deque([], maxlen=6)


    def buffer(self, dt):
        """Buffer the state and timestep. Dynamically precompute 
        the variable timestep BDF coefficients on the fly for the 
        current timestep.
        
        Parameters
        ----------
        dt : float
            integration timestep
        """

        #reset optimizer
        self.opt.reset()
    
        #add to histories (solution and timestep)            
        self.history.appendleft(self.x)
        self.history_dt.appendleft(dt)

        #flag for startup method
        self._needs_startup = len(self.history) < 6

        #buffer with startup method
        if self._needs_startup:
            self.startup.buffer(dt)

        #precompute coefficients here, where buffers are available
        self.F, self.K = {}, {}
        for n, _ in enumerate(self.history_dt, 1):
            self.F[n], self.K[n] = compute_bdf_coefficients(n, np.array(self.history_dt))


    # methods for adaptive timestep solvers --------------------------------------------

    def error_controller(self, tr_m, tr_p):
        """Compute scaling factor for adaptive timestep based on absolute and 
        relative tolerances of the local truncation error estimate obtained from 
        esimated lower and higher order solution. 

        Checks if the error tolerance is achieved and returns a success metric.

        Adapts the stepping order such that the normalized error is minimized and 
        larger steps can be taken by the integrator.

        Parameters
        ----------
        tr_m : array[float]
            lower order truncation error estimate
        tr_p : array[float]
            higher order truncation error estimate

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #compute scaling factors (avoid division by zero)
        scale = self.tolerance_lte_abs + self.tolerance_lte_rel * np.abs(self.x)

        #compute scaled truncation error (element-wise)
        scaled_error_m = np.abs(tr_m) / scale
        scaled_error_p = np.abs(tr_p) / scale

        #compute the error norm and clip it
        error_norm_m = np.clip(float(np.max(scaled_error_m)), TOLERANCE, None)
        error_norm_p = np.clip(float(np.max(scaled_error_p)), TOLERANCE, None)      

        #success metric (use lower order estimate)
        success = error_norm_m <= 1.0

        #compute timestep scale factor using accuracy order of truncation error
        timestep_rescale = self.beta / error_norm_m ** (1/self.n)  

        #clip the rescale factor to a reasonable range
        timestep_rescale = np.clip(timestep_rescale, SOL_SCALE_MIN, SOL_SCALE_MAX)

        #decrease the order if smaller order is more accurate (stability)
        if error_norm_m < error_norm_p:
            self.n = max(self.n-1, self.n_min)
        
        #increase the order if larger order is more accurate (accuracy -> larger steps)
        else:
            self.n = min(self.n+1, self.n_max)

        return success, error_norm_p, timestep_rescale


    # methods for timestepping ---------------------------------------------------------

    def solve(self, f, J, dt):
        """Solves the implicit update equation using the optimizer of the engine.
        
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

        #not enough history for full order -> solve with startup method
        if self._needs_startup:
            err = self.startup.solve(f, J, dt)
            self.x = self.startup.get()
            return err
        
        #fixed-point function update (faster then sum comprehension)
        g = self.F[self.n] * dt * f
        for b, k in zip(self.history, self.K[self.n]):
            g = g + b * k

        #use the jacobian
        if J is not None:

            #optimizer step with block local jacobian
            self.x, err = self.opt.step(self.x, g, self.F[self.n] * dt * J)

        else:
            #optimizer step (pure)
            self.x, err = self.opt.step(self.x, g, None)

        #return the fixed-point residual
        return err


    def step(self, f, dt):
        """Finalizes the timestep by resetting the solver for the implicit 
        update equation and computing the lower and higher order estimate 
        of the solution. 

        Then calls the error controller.

        Parameters
        ----------
        f : array_like
            evaluation of function
        dt : float 
            integration timestep

        Returns 
        -------
        success : bool
            True if the timestep was successful
        error : float
            estimated error of the internal error controller
        scale : float
            estimated timestep rescale factor for error control
        """

        #not enough history for full order -> step with startup method
        if self._needs_startup:
            suc, err, scl = self.startup.step(f, dt)
            self.x = self.startup.get()
            return suc, err, scl

        #lower and higher order
        n_m, n_p = self.n - 1, self.n + 1 

        #estimate truncation error from lower order solution
        tr_m = self.x - self.F[n_m] * dt * f
        for b, k in zip(self.history, self.K[n_m]):
            tr_m = tr_m - b * k

        #estimate truncation error from higher order solution
        tr_p = self.x - self.F[n_p] * dt * f
        for b, k in zip(self.history, self.K[n_p]):
            tr_p = tr_p - b * k

        return self.error_controller(tr_m, tr_p)
