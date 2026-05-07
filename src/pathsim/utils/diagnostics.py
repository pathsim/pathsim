#########################################################################################
##
##                     CONVERGENCE TRACKING AND DIAGNOSTICS
##                            (utils/diagnostics.py)
##
##         Convergence tracker classes for the simulation solver loops
##         and optional per-timestep diagnostics snapshot.
##
#########################################################################################

# IMPORTS ===============================================================================

from dataclasses import dataclass, field


# CONVERGENCE TRACKER ===================================================================

class ConvergenceTracker:
    """Tracks per-object scalar errors and convergence for fixed-point loops.

    Used by the algebraic loop solver (keyed by ConnectionBooster) and
    the implicit ODE solver (keyed by Block).

    Attributes
    ----------
    errors : dict
        object -> float, per-object error from most recent iteration
    max_error : float
        maximum error across all objects in current iteration
    iterations : int
        number of iterations taken
    """

    __slots__ = ('errors', 'max_error', 'iterations')

    def __init__(self):
        self.errors = {}
        self.max_error = 0.0
        self.iterations = 0


    def reset(self):
        """Clear all state."""
        self.errors.clear()
        self.max_error = 0.0
        self.iterations = 0


    def begin_iteration(self):
        """Reset per-iteration state before sweeping objects."""
        self.errors.clear()
        self.max_error = 0.0


    def record(self, obj, error):
        """Record a single object's error and update the running max."""
        self.errors[obj] = error
        if error > self.max_error:
            self.max_error = error


    def converged(self, tolerance):
        """Check if max error is within tolerance."""
        return self.max_error <= tolerance


    def details(self, label_fn):
        """Format per-object error breakdown for error messages.

        Parameters
        ----------
        label_fn : callable
            obj -> str, produces a human-readable label

        Returns
        -------
        list[str]
            formatted lines like "  Integrator: 1.23e-04"
        """
        return [f"  {label_fn(obj)}: {err:.2e}" for obj, err in self.errors.items()]


# STEP TRACKER ==========================================================================

class StepTracker:
    """Tracks per-block adaptive step results.

    Used by the adaptive error control loop. Each block produces a tuple
    (success, err_norm, scale) and this tracker aggregates them.

    Attributes
    ----------
    errors : dict
        block -> (success, err_norm, scale) from most recent step
    success : bool
        AND of all block successes
    max_error : float
        maximum error norm across all blocks
    min_scale : float | None
        minimum scale factor (None if no block provides one)
    """

    __slots__ = ('errors', 'success', 'max_error', 'min_scale')

    def __init__(self):
        self.errors = {}
        self.success = True
        self.max_error = 0.0
        self.min_scale = None


    def reset(self):
        """Clear state for a new step."""
        self.errors.clear()
        self.success = True
        self.max_error = 0.0
        self.min_scale = None


    def record(self, block, success, err_norm, scale):
        """Record a single block's step result."""
        self.errors[block] = (success, err_norm, scale)
        if not success:
            self.success = False
        if err_norm > self.max_error:
            self.max_error = err_norm
        if scale is not None:
            if self.min_scale is None or scale < self.min_scale:
                self.min_scale = scale


    @property
    def scale(self):
        """Effective scale factor (1.0 when no block provides one)."""
        return self.min_scale if self.min_scale is not None else 1.0


# DIAGNOSTICS SNAPSHOT ==================================================================

@dataclass
class Diagnostics:
    """Per-timestep convergence diagnostics snapshot.

    Populated by the simulation engine after each successful timestep
    from the three convergence trackers. Provides read-only accessors
    for the worst offending block or connection.

    Attributes
    ----------
    time : float
        simulation time
    loop_residuals : dict
        per-booster algebraic loop residuals (booster -> residual)
    loop_iterations : int
        number of algebraic loop iterations taken
    solve_residuals : dict
        per-block implicit solver residuals (block -> residual)
    solve_iterations : int
        number of implicit solver iterations taken
    step_errors : dict
        per-block adaptive step data (block -> (success, err_norm, scale))
    """
    time: float = 0.0
    loop_residuals: dict = field(default_factory=dict)
    loop_iterations: int = 0
    solve_residuals: dict = field(default_factory=dict)
    solve_iterations: int = 0
    step_errors: dict = field(default_factory=dict)


    @staticmethod
    def _label(obj):
        """Human-readable label for a block or booster."""
        if hasattr(obj, 'connection'):
            return str(obj.connection)
        return obj.__class__.__name__


    def worst_block(self):
        """Block with the highest residual across solve and step errors.

        Returns
        -------
        tuple[str, float] or None
            (label, error) or None if no data
        """
        worst, worst_err = None, -1.0

        for obj, err in self.solve_residuals.items():
            if err > worst_err:
                worst, worst_err = obj, err

        for obj, (_, err_norm, _) in self.step_errors.items():
            if err_norm > worst_err:
                worst, worst_err = obj, err_norm

        if worst is None:
            return None
        return self._label(worst), worst_err


    def worst_booster(self):
        """Connection booster with the highest algebraic loop residual.

        Returns
        -------
        tuple[str, float] or None
            (label, residual) or None if no data
        """
        if not self.loop_residuals:
            return None

        worst = max(self.loop_residuals, key=self.loop_residuals.get)
        return self._label(worst), self.loop_residuals[worst]


    def summary(self):
        """Formatted summary of this diagnostics snapshot.

        Returns
        -------
        str
            human-readable diagnostics summary
        """
        lines = [f"Diagnostics at t = {self.time:.6f}"]

        if self.step_errors:
            lines.append(f"\n  Adaptive step errors:")
            for obj, (suc, err, scl) in self.step_errors.items():
                status = "OK" if suc else "FAIL"
                scl_str = f"{scl:.3f}" if scl is not None else "N/A"
                lines.append(f"    {status}  {self._label(obj)}: err={err:.2e}, scale={scl_str}")

        if self.solve_residuals:
            lines.append(f"\n  Implicit solver residuals ({self.solve_iterations} iterations):")
            for obj, err in self.solve_residuals.items():
                lines.append(f"    {self._label(obj)}: {err:.2e}")

        if self.loop_residuals:
            lines.append(f"\n  Algebraic loop residuals ({self.loop_iterations} iterations):")
            for obj, err in self.loop_residuals.items():
                lines.append(f"    {self._label(obj)}: {err:.2e}")

        return "\n".join(lines)
