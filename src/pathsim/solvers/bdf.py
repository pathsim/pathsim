########################################################################################
##
##                         BACKWARD DIFFERENTIATION FORMULAS
##                                 (solvers/bdf.py)
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np

from collections import deque

from ._solver import ImplicitSolver
from .dirk3 import DIRK3


# BASE BDF SOLVER ======================================================================

class BDF(ImplicitSolver):
    """Base class for the backward differentiation formula (BDF) integrators.
    
    Notes
    ----- 
    This solver class is not intended to be used directly

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
    startup : Solver
        internal solver instance for startup (building history) 
        of multistep methods (using 'DIRK3' for 'BDF' methods)
    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #integration order
        self.n = None

        #bdf coefficients for orders 1 to 6
        self.K = {
            1:[1.0], 
            2:[4/3, -1/3], 
            3:[18/11, -9/11, 2/11], 
            4:[48/25, -36/25, 16/25, -3/25],
            5:[300/137, -300/137, 200/137, -75/137, 12/137],
            6:[360/147, -450/147, 400/147, -225/147, 72/147, -10/147]
            }
        self.F = {1:1.0, 2:2/3, 3:6/11, 4:12/25, 5:60/137, 6:60/147}

        #initialize startup solver from 'self' and flag
        self._needs_startup = True
        self.startup = DIRK3.cast(self, self.parent)


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
        engine : BDF
            instance of `BDF` solver with params and state from `other`
        """
        engine = super().cast(other, parent, **solver_kwargs)
        engine.startup = DIRK3.cast(engine, parent)

        return engine


    @classmethod
    def create(cls, initial_value, parent=None, from_engine=None, **solver_kwargs):
        """Create a new BDF solver, properly initializing the startup solver.

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
        engine : BDF
            new BDF solver instance
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
            engine.startup = DIRK3.create(initial_value, parent, **solver_kwargs)
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


    def reset(self):
        """"Resets integration engine to initial state."""

        #clear history (BDF solution buffer)
        self.history.clear()

        #overwrite state with initial value (ensure array format)
        self.x = np.atleast_1d(self.initial_value).copy()

        #reset startup solver
        self.startup.reset()


    def buffer(self, dt):
        """buffer the state for the multistep method
        
        Parameters
        ----------
        dt : float
            integration timestep
        """
            
        #reset optimizer
        self.opt.reset()

        #add current solution to history
        self.history.appendleft(self.x)

        #flag for startup method, not enough history
        self._needs_startup = len(self.history) < self.n

        #buffer with startup method
        if self._needs_startup:
            self.startup.buffer(dt)


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

        #fixed-point function update
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
        """Performs the explicit timestep for (t+dt) based 
        on the state and input at (t).

        Note
        ----
        This is only required for the startup solver.

        Parameters
        ----------
        f : numeric, array[numeric]
            evaluation of rhs function
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

        #not enough histors -> step the startup solver
        if self._needs_startup:
            self.startup.step(f, dt)
            self.x = self.startup.get()

        return True, 0.0, None


# SOLVERS ==============================================================================

class BDF2(BDF):
    """Fixed-step 2nd order BDF method. A-stable.

    .. math::

        x_{n+1} = \\tfrac{4}{3}\\,x_n - \\tfrac{1}{3}\\,x_{n-1}
                  + \\tfrac{2}{3}\\,h\\,f(x_{n+1}, t_{n+1})

    Uses ``DIRK3`` as startup solver for the first step.

    Characteristics
    ---------------
    * Order: 2
    * Implicit linear multistep, fixed timestep
    * A-stable

    Note
    ----
    The workhorse fixed-step stiff solver. A-stability means no eigenvalue
    in the left half-plane causes instability, regardless of the timestep.
    Well suited for block diagrams with a fixed simulation clock and
    moderately-to-very stiff dynamics. For adaptive stepping, use ``GEAR21``
    or ``ESDIRK43``.

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

        #integration order (local)
        self.n = 2

        #longer history for BDF
        self.history = deque([], maxlen=2)


class BDF3(BDF):
    """Fixed-step 3rd order BDF method. :math:`A(\\alpha)`-stable with
    :math:`\\alpha \\approx 86°`.

    Uses ``DIRK3`` as startup solver for the first two steps.

    Characteristics
    ---------------
    * Order: 3
    * Implicit linear multistep, fixed timestep
    * :math:`A(\\alpha)`-stable, :math:`\\alpha \\approx 86°`

    Note
    ----
    Higher accuracy than ``BDF2`` with only a slight reduction in the
    stability wedge. The :math:`86°` sector covers nearly the entire left
    half-plane, so most stiff block diagrams remain well-handled. For
    adaptive stepping, use ``GEAR32``.

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

        #integration order (local)
        self.n = 3

        #longer history for BDF
        self.history = deque([], maxlen=3)


class BDF4(BDF):
    """Fixed-step 4th order BDF method. :math:`A(\\alpha)`-stable with
    :math:`\\alpha \\approx 73°`.

    Uses ``DIRK3`` as startup solver for the first three steps.

    Characteristics
    ---------------
    * Order: 4
    * Implicit linear multistep, fixed timestep
    * :math:`A(\\alpha)`-stable, :math:`\\alpha \\approx 73°`

    Note
    ----
    The narrower stability wedge compared to ``BDF2``/``BDF3`` means
    eigenvalues close to the imaginary axis may be poorly damped. Safe for
    block diagrams whose stiff modes are strongly dissipative (well inside
    the left half-plane). For problems with near-imaginary eigenvalues (e.g.
    lightly damped oscillators), prefer lower-order BDF or an ESDIRK solver.

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

        #integration order (local)
        self.n = 4

        #longer history for BDF
        self.history = deque([], maxlen=4)


class BDF5(BDF):
    """Fixed-step 5th order BDF method. :math:`A(\\alpha)`-stable with
    :math:`\\alpha \\approx 51°`.

    Uses ``DIRK3`` as startup solver for the first four steps.

    Characteristics
    ---------------
    * Order: 5
    * Implicit linear multistep, fixed timestep
    * :math:`A(\\alpha)`-stable, :math:`\\alpha \\approx 51°`

    Note
    ----
    The stability wedge is noticeably smaller than ``BDF3`` or ``BDF4``.
    Only appropriate when the stiff eigenvalues of the block diagram are
    concentrated well inside the left half-plane and high accuracy per step
    is essential. For most stiff systems, ``BDF2`` or ``BDF3`` with a
    smaller timestep is more robust.

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

        #integration order (local)
        self.n = 5

        #longer history for BDF
        self.history = deque([], maxlen=5)


class BDF6(BDF):
    """Fixed-step 6th order BDF method. **Not** A-stable
    (:math:`\\alpha \\approx 18°`).

    Uses ``DIRK3`` as startup solver for the first five steps.

    Characteristics
    ---------------
    * Order: 6
    * Implicit linear multistep, fixed timestep
    * :math:`A(\\alpha)`-stable, :math:`\\alpha \\approx 18°` (not A-stable)

    Note
    ----
    The very narrow stability wedge means that most stiff problems will be
    unstable at practical timestep sizes. Provided mainly for completeness.
    For 6th order accuracy on non-stiff systems, the explicit ``RKV65`` is
    cheaper. For stiff systems, ``BDF2``--``BDF4`` with a smaller timestep
    or an ESDIRK solver are far more robust.

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

        #integration order (local)
        self.n = 6

        #longer history for BDF
        self.history = deque([], maxlen=6)