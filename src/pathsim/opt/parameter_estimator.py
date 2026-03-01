#########################################################################################
##
##                  PARAMETER ESTIMATION TOOLKIT (PathSim Extension Layer)
##                              (parameter_estimator.py)
##
##                                  Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

import copy
import types as _types
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import scipy.optimize as sci_opt

from ..blocks._block import Block
from ..utils.timeseries_data import TimeSeriesData

# Sentinel for "argument not provided" — distinguishes None from unset
_UNSET = object()


# DEEP-COPY HELPERS =====================================================================
#
# Python treats functions (including lambdas) as atomic during deepcopy: the
# new Operator object holds the exact same function object, so any closure that
# captures `self` (e.g. `lambda x: x * self.gain` in Amplifier) still refers
# to the ORIGINAL block after the copy.  The helpers below fix this by walking
# all block operators after deepcopy and rebuilding any function whose closure
# contains an original-block reference.

def _make_cell(value):
    """Return a new closure cell containing *value*."""
    def _factory():
        return value
    return _factory.__closure__[0]


def _rebind_func_closure(func, memo):
    """Return a copy of *func* with stale closure cells rebound via *memo*.

    After ``copy.deepcopy(sim, memo)``, *memo* maps ``id(original) -> copy``.
    Any closure cell whose content appears as a key in *memo* (and whose copy
    differs from the original) is replaced.  Returns *func* unchanged when no
    rebinding is required.
    """
    closure = getattr(func, "__closure__", None)
    if not closure:
        return func

    new_cells = []
    needs_fix = False
    for cell in closure:
        try:
            obj = cell.cell_contents
        except ValueError:            # empty cell
            new_cells.append(cell)
            continue
        copied = memo.get(id(obj))
        if copied is not None and copied is not obj:
            new_cells.append(_make_cell(copied))
            needs_fix = True
        else:
            new_cells.append(cell)

    if not needs_fix:
        return func

    new_func = _types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        tuple(new_cells),
    )
    if func.__kwdefaults__ is not None:
        new_func.__kwdefaults__ = func.__kwdefaults__
    return new_func


def _fix_sim_operator_closures(sim_copy, memo):
    """Walk every block operator in *sim_copy* and rebind stale closures.

    Covers ``op_alg`` (``Operator``) and ``op_dyn`` (``DynamicOperator``),
    checking all stored callable attributes: ``_func``, ``_jac``,
    ``_jac_x``, ``_jac_u``.
    """
    for block in getattr(sim_copy, "blocks", []):
        for op_attr in ("op_alg", "op_dyn"):
            op = getattr(block, op_attr, None)
            if op is None:
                continue
            for func_attr in ("_func", "_jac", "_jac_x", "_jac_u"):
                func = getattr(op, func_attr, None)
                if func is None or not callable(func):
                    continue
                fixed = _rebind_func_closure(func, memo)
                if fixed is not func:
                    setattr(op, func_attr, fixed)


def _deepcopy_sim(simulator):
    """Deep-copy *simulator* and fix operator closure references.

    Equivalent to ``copy.deepcopy(simulator)`` but additionally walks every
    block's algebraic and dynamic operators to rebind any lambda closure that
    still references an original block object after the copy.
    """
    memo: dict = {}
    sim_copy = copy.deepcopy(simulator, memo)
    _fix_sim_operator_closures(sim_copy, memo)
    return sim_copy


__all__ = [
    "Parameter",
    "BlockParameter",
    "FreeParameter",
    "SharedBlockParameter",
    "ScopeSignal",
    "SimRunner",
    "Experiment",
    "ParameterEstimator",
    "EstimatorResult",
]


# PARAMETER DECLARATION =================================================================

class Parameter:
    """Unified estimation parameter.

    A parameter is either free (used directly in model code) or block-bound
    (mapped to a PathSim block attribute). Optional transforms allow the
    optimizer to work in an unconstrained space while applying a physically
    meaningful value to the model (e.g. ``np.exp`` to enforce positivity).

    Parameters
    ----------
    name : str
        Parameter identifier.
    value : float
        Initial value in optimizer space.
    bounds : tuple[float, float]
        Lower / upper bounds in optimizer space.
    transform : callable, optional
        Applied when the parameter is read: ``model_value = transform(optimizer_value)``.
    block : object, optional
        Target block for block-bound parameters.
    attribute : str, optional
        Dotted attribute path on *block* (e.g. ``"config.gain"``).

    Notes
    -----
    Calling a ``Parameter`` instance (``p()``) returns the model-space value
    after the optional transform.  ``p.value`` always returns the optimizer-space
    value.  For block-bound parameters, ``set()`` immediately writes the
    transformed value to the target attribute.

    Example
    -------
    .. code-block:: python

        # Free parameter with log-scale transform
        k = Parameter("k", value=2.0, bounds=(0, 4), transform=np.exp)
        k()       # model-space value: exp(2.0) ≈ 7.39
        k.value   # optimizer-space value: 2.0

        # Block-bound parameter
        p = Parameter("gain", value=5.0, block=amp, attribute="gain")
        p.set(10.0)   # also sets amp.gain = 10.0
    """

    def __init__(
        self,
        name: str,
        value: float = 1.0,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        transform: Callable[[float], float] | None = None,
        block: Any | None = None,
        attribute: str | None = None,
    ):
        self.name = name
        self._value = float(value)
        self.transform = transform
        self.block = block
        self.attribute = attribute

        if block is not None and attribute is None:
            raise ValueError("attribute must be provided when block is specified")

        lo, hi = bounds
        if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
            raise ValueError(
                f"Parameter '{name}': lower bound {lo} > upper bound {hi}"
            )
        self.bounds = bounds

        if np.isfinite(lo) and float(value) < lo:
            warnings.warn(
                f"Parameter '{name}': initial value {value} < lower bound {lo}",
                UserWarning,
                stacklevel=2,
            )
        if np.isfinite(hi) and float(value) > hi:
            warnings.warn(
                f"Parameter '{name}': initial value {value} > upper bound {hi}",
                UserWarning,
                stacklevel=2,
            )

        self._is_block_param = block is not None
        self.set(value)


    @property
    def is_block_parameter(self) -> bool:
        """True if this parameter is bound to a block attribute."""
        return self._is_block_param


    @property
    def is_free_parameter(self) -> bool:
        """True if this is a free (non-block) parameter."""
        return not self._is_block_param


    @property
    def value(self) -> float:
        """Current optimizer-space value."""
        return self._value


    @value.setter
    def value(self, new_value: float) -> None:
        self.set(new_value)


    def __call__(self) -> float:
        """Return the model-space value (after optional transform)."""
        return self.transform(self._value) if self.transform is not None else self._value


    def set(self, value: float) -> None:
        """Set the optimizer-space value and push to the target block if bound."""
        self._value = float(value)
        transformed = self()

        if self._is_block_param:
            obj = self.block
            attrs = self.attribute.split(".")
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], transformed)


    def __repr__(self) -> str:
        if self._is_block_param:
            return (
                f"Parameter(name={self.name!r}, value={self._value}, "
                f"block={type(self.block).__name__}, "
                f"attribute={self.attribute!r}, bounds={self.bounds})"
            )
        return (
            f"Parameter(name={self.name!r}, value={self._value}, "
            f"bounds={self.bounds})"
        )


def BlockParameter(block, attribute, name=None, **kwargs):
    """Factory for block-bound Parameters.

    Parameters
    ----------
    block : object
        Target block.
    attribute : str
        Attribute path (e.g. ``"value"`` or ``"config.gain"``).
    name : str, optional
        Parameter name; defaults to ``"{BlockType}.{attribute}"``.

    Returns
    -------
    Parameter
    """
    if name is None:
        name = f"{type(block).__name__}.{attribute}"
    return Parameter(name=name, block=block, attribute=attribute, **kwargs)


def FreeParameter(name, **kwargs):
    """Factory for free (non-block) Parameters.

    Parameters
    ----------
    name : str
        Parameter identifier.
    **kwargs
        Forwarded to :class:`Parameter`.

    Returns
    -------
    Parameter
    """
    return Parameter(name=name, **kwargs)


class SharedBlockParameter(Parameter):
    """A block-bound parameter applied to the same attribute on multiple blocks.

    Intended for multi-experiment fitting where each experiment uses a
    deep-copied simulation, and the fitted parameter should be shared globally.

    Parameters
    ----------
    name : str
        Parameter identifier.
    targets : list[object]
        Block instances that all receive the same transformed value on ``set()``.
    attribute : str
        Attribute path applied to every target.
    value : float
        Initial optimizer-space value.
    bounds : tuple[float, float]
        Lower / upper bounds in optimizer space.
    transform : callable, optional
        Applied when the parameter is pushed to targets.

    Notes
    -----
    Initialisation bypasses ``Parameter.__init__`` to avoid the fragile
    ordering constraint that ``self.targets`` must exist before ``set()`` is
    dispatched.  All parent attributes are initialised explicitly so the full
    ``Parameter`` interface is preserved.
    """

    def __init__(
        self,
        name: str,
        targets: list[Any],
        attribute: str,
        value: float = 1.0,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        transform: Callable[[float], float] | None = None,
    ):
        if not targets:
            raise ValueError("targets must be a non-empty list")

        lo, hi = bounds
        if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
            raise ValueError(
                f"SharedBlockParameter '{name}': lower bound {lo} > upper bound {hi}"
            )

        # Initialise all Parameter attributes directly — do NOT call
        # super().__init__() because it would dispatch to our set() before
        # self.targets is assigned, causing an AttributeError.
        self.name = name
        self._value = 0.0          # placeholder; real value written by set() below
        self.bounds = bounds
        self.transform = transform
        self.block = targets[0]    # kept for repr / debug parity with Parameter
        self.attribute = attribute
        self._is_block_param = True
        self.targets = targets

        if np.isfinite(lo) and float(value) < lo:
            warnings.warn(
                f"SharedBlockParameter '{name}': initial value {value} < lower bound {lo}",
                UserWarning,
                stacklevel=2,
            )
        if np.isfinite(hi) and float(value) > hi:
            warnings.warn(
                f"SharedBlockParameter '{name}': initial value {value} > upper bound {hi}",
                UserWarning,
                stacklevel=2,
            )

        self.set(value)


    def set(self, value: float) -> None:
        """Set optimizer value and push to all target blocks."""
        self._value = float(value)
        transformed = self()

        for tgt in self.targets:
            obj = tgt
            attrs = self.attribute.split(".")
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], transformed)


# SCOPE SIGNAL ==========================================================================

@dataclass
class ScopeSignal:
    """Scope output selector.

    Reads a single port from an object that provides ``read() -> (t, y)``.

    Parameters
    ----------
    scope : object, optional
        Scope-like provider with ``.read()``.
    port : int
        Port index to extract from multi-port scopes.
    block_type : str, optional
        Block class name for deferred resolution (multi-experiment support).
    occurrence : int, optional
        Index of the block in the type-filtered list (multi-experiment support).

    Notes
    -----
    When ``scope`` is ``None``, the signal must be resolved via
    :meth:`ParameterEstimator._resolve_output` before reading. This allows the
    same signal specification to be re-bound to the matching deep-copied scope
    in each experiment's simulation.
    """

    scope: object | None = None
    port: int = 0
    block_type: str | None = None
    occurrence: int | None = None


    def _extract(self, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract the selected port from raw scope output arrays.

        Parameters
        ----------
        t : array_like
            Time vector.
        y : array_like
            Output array, shape ``(n_time,)`` or ``(n_ports, n_time)``.
            If unambiguously ``(n_time, n_ports)`` (time-first), it is transposed.
        """
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 1:
            return t_arr, y_arr.reshape(-1)

        if y_arr.ndim == 2:
            n_rows, n_cols = y_arr.shape

            if n_rows == t_arr.size and n_cols != t_arr.size:
                # Unambiguously (n_time, n_ports) → transpose to (n_ports, n_time)
                y_arr = y_arr.T
            elif n_rows == t_arr.size and n_cols == t_arr.size:
                # Square array: cannot determine time axis automatically.
                raise ValueError(
                    f"ScopeSignal._extract: ambiguous square scope output shape "
                    f"{y_arr.shape} with {t_arr.size} time samples. "
                    "Ensure the scope records data in (n_ports, n_time) layout "
                    "(the default PathSim Scope layout)."
                )
            # Assumed (n_ports, n_time) from here
            n_ports = y_arr.shape[0]
            if self.port >= n_ports:
                raise IndexError(
                    f"ScopeSignal port={self.port} out of range; "
                    f"scope output has {n_ports} port(s)"
                )
            return t_arr, y_arr[self.port, :].reshape(-1)

        raise ValueError("ScopeSignal supports 1D or 2D scope data only")


    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read ``(t, y)`` from the scope and extract the selected port."""
        if self.scope is None:
            raise ValueError(
                "ScopeSignal.scope is None; it must be resolved before reading"
            )
        t, y = self.scope.read()
        return self._extract(t, y)


# SIMULATION RUNNER =====================================================================

@dataclass
class SimRunner:
    """Simulation runner adapter.

    Provides a reset-and-run interface suitable for repeated evaluations inside
    an optimization loop.

    Parameters
    ----------
    sim : object
        PathSim simulation with ``.run()`` and ``.reset()`` methods.
    output : ScopeSignal, optional
        Primary output extractor (used when the runner is called as a callable).
    duration : float
        Simulation duration per evaluation.
    post_reset_hooks : list[callable], optional
        Callables executed immediately after ``sim.reset()``.
    pre_run : callable, optional
        Callable executed just before ``sim.run()``.
    adaptive : bool
        Passed as ``adaptive=`` to ``sim.run()``.
    reset_time : float
        Time passed into ``sim.reset(time=...)``.
    suppress_reset_log : bool
        Temporarily disable sim logging during reset/run for cleaner output.
    """

    sim: object
    output: ScopeSignal | None
    duration: float
    post_reset_hooks: list[Callable[[], None]] | None = None
    pre_run: Callable[[], None] | None = None
    adaptive: bool = False
    reset_time: float = 0.0
    suppress_reset_log: bool = True


    def after_reset(self, fn: Callable[[], None]) -> "SimRunner":
        """Register a callable executed after every ``sim.reset()``."""
        if self.post_reset_hooks is None:
            self.post_reset_hooks = []
        self.post_reset_hooks.append(fn)
        return self


    def before_run(self, fn: Callable[[], None]) -> "SimRunner":
        """Set a callable executed immediately before ``sim.run()``."""
        self.pre_run = fn
        return self


    def run(self) -> None:
        """Reset and run the simulation."""
        old_log = None
        if self.suppress_reset_log and hasattr(self.sim, "log"):
            old_log = self.sim.log
            self.sim.log = False

        try:
            self.sim.reset(time=self.reset_time)

            if self.post_reset_hooks is not None:
                for hook in self.post_reset_hooks:
                    hook()

            if self.pre_run is not None:
                self.pre_run()

            self.sim.run(duration=self.duration, reset=False, adaptive=self.adaptive)

        finally:
            if old_log is not None:
                self.sim.log = old_log


# EXPERIMENT ============================================================================

@dataclass
class Experiment:
    """A single experiment with one or more measurement–output pairs.

    Each experiment is evaluated by running its ``runner`` once per objective
    call, then comparing each measurement against its mapped scope output.

    Parameters
    ----------
    runner : SimRunner
        Runner responsible for resetting and running the simulation.
    duration : float, optional
        Explicit duration override; otherwise derived from measurements.
    measurements : list[TimeSeriesData]
        Measurement datasets for this experiment.
    outputs : list[ScopeSignal]
        Scope output selectors corresponding to each dataset (same order).
    sigma : list[float | None]
        Per-dataset noise scaling for residual normalization.
    """

    runner: Any
    duration: float | None = None
    measurements: list[TimeSeriesData] | None = None
    outputs: list[ScopeSignal] | None = None
    sigma: list[float | None] | None = None


    def __post_init__(self) -> None:
        if self.measurements is None:
            self.measurements = []
        if self.outputs is None:
            self.outputs = []
        if self.sigma is None:
            self.sigma = []


# ESTIMATOR RESULT ======================================================================

@dataclass
class EstimatorResult:
    """Parameter estimation result container."""

    x: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str


    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"EstimatorResult({status}, cost={self.cost:.4g}, "
            f"nfev={self.nfev}, x={self.x})"
        )


# PARAMETER ESTIMATOR ===================================================================

class ParameterEstimator:
    """Parameter estimation driver.

    Manages parameters, simulation experiments, and measurement data; drives
    SciPy-based fitting via a residual objective.

    Parameters
    ----------
    simulator : object, optional
        PathSim simulation (must have ``.run()`` and ``.reset()``).
        If provided, a default experiment is created automatically.
    parameters : list[Parameter], optional
        Free or block-bound parameters added as global parameters on construction.
    duration : float, optional
        Default simulation duration for the first experiment. Overridden by
        measurement time extent when measurements are added.
    adaptive : bool
        Enable adaptive (variable) step-size control when running the
        simulation.  Pass ``True`` if the solver is configured with error
        tolerances (``tolerance_lte_rel`` / ``tolerance_lte_abs``); use
        ``False`` (default) for fixed-step simulations — adaptive mode
        on a fixed-step solver has no effect and adds overhead.
        This value is inherited by subsequent :meth:`add_experiment` calls
        unless explicitly overridden.
    pre_run : callable, optional
        Callable executed just before each ``sim.run()`` in the default
        experiment.  Useful for updating time-varying inputs or resetting
        auxiliary state between optimizer evaluations.
        Inherited by subsequent :meth:`add_experiment` calls unless overridden.

    Notes
    -----
    **Choosing how to register parameters**

    - ``add_block_parameter(block, "attr", ...)`` — simplest approach;
      directly writes the fitted value to a block attribute each iteration.
      Use this for single-experiment problems.
    - ``add_global_block_parameter("BlockType", "attr", ...)`` — same idea
      but syncs the value across all deep-copied experiment simulations.
      Use this for multi-experiment problems.
    - ``add_parameters([p])`` / ``parameters=[p]`` constructor arg — for
      *free* parameters that are read via ``p()`` inside model closures
      rather than bound to a specific block attribute.

    **Common usage patterns**

    *Pattern 1 — Single experiment, block-bound parameters:*

    .. code-block:: python

        est = ParameterEstimator(simulator=sim, adaptive=True)
        est.add_block_parameter(amp, "gain", value=1.0, bounds=(0, 10))
        est.add_timeseries(meas, signal=scope[0], sigma=1.0)
        result = est.fit()
        est.display()

    *Pattern 2 — Multi-experiment with a shared (global) parameter:*

    .. code-block:: python

        est = ParameterEstimator(simulator=sim, adaptive=True)
        est.add_experiment(sim, copy_sim=True)   # independent copy for exp 1
        est.add_global_block_parameter("Gain", "value", value=1.0, bounds=(0, 10))
        est.add_timeseries(meas0, signal=scope[0], sigma=1.0, experiment=0)
        est.add_timeseries(meas1, signal=scope[0], sigma=1.0, experiment=1)
        result = est.fit()

    *Pattern 3 — Free parameters used directly in model closures:*

    .. code-block:: python

        k = Parameter("k", value=1.0, bounds=(0, 5), transform=np.exp)

        def ode_rhs(x, u, t):
            return -k() * x       # k() returns exp(k.value)

        est = ParameterEstimator(simulator=sim, parameters=[k])
        est.add_timeseries(meas, signal=scope[0])
        result = est.fit()

    Example
    -------
    .. code-block:: python

        from pathsim.opt import ParameterEstimator, TimeSeriesData

        est = ParameterEstimator(simulator=sim, adaptive=True)
        est.add_block_parameter(block, "gain", value=1.0, bounds=(0, 10))
        est.add_timeseries(meas, signal=scope[0], sigma=1.0)
        result = est.fit(max_nfev=100)
        est.display()
    """

    def __init__(
        self,
        simulator=None,
        *,
        parameters: list[Parameter] | None = None,
        duration: float | None = None,
        adaptive: bool = False,
        pre_run: Callable | None = None,
    ):
        # Global parameters (applied to every experiment)
        self.global_parameters: list[Parameter] = list(parameters or [])
        # Per-experiment parameters (applied only to their own experiment)
        self.local_parameters: list[list[Parameter]] = []
        # Experiments
        self.experiments: list[Experiment] = []

        # Cache for the flattened parameter list — invalidated on any add_*
        self._params_cache: list[Parameter] | None = None
        # Cache for the last residuals evaluation
        self._cached_x: np.ndarray | None = None
        self._cached_residuals: np.ndarray | None = None
        # Deferred duration update — set True when measurements/experiments change
        self._duration_dirty: bool = False
        self._duration: float = 0.0

        # Reference simulator for cloning additional experiments
        self._base_simulator = (
            simulator if simulator is not None and hasattr(simulator, "run") else None
        )
        # Store constructor defaults so add_experiment() can inherit them
        self._default_adaptive: bool = adaptive
        self._default_pre_run: Callable | None = pre_run
        # Keep legacy dict for _ensure_experiment auto-cloning
        self._default_runner_kwargs = dict(adaptive=adaptive, pre_run=pre_run)

        if simulator is not None:
            self.add_experiment(
                simulator,
                duration=duration,
                adaptive=adaptive,
                pre_run=pre_run,
            )


    # PROPERTIES ------------------------------------------------------------------------

    @property
    def parameters(self) -> list[Parameter]:
        """Flattened parameter list: globals first, then locals experiment-by-experiment."""
        if self._params_cache is None:
            params: list[Parameter] = list(self.global_parameters)
            for exp_params in self.local_parameters:
                params.extend(exp_params)
            self._params_cache = params
        return self._params_cache


    def _invalidate_cache(self) -> None:
        """Invalidate parameter list and residuals caches."""
        self._params_cache = None
        self._cached_x = None
        self._cached_residuals = None


    @property
    def runners(self) -> list[SimRunner]:
        """Live list of runners, one per experiment."""
        return [exp.runner for exp in self.experiments]


    @property
    def duration(self) -> float:
        """Total measurement time extent across all experiments.

        Triggers a lazy duration update if measurements have been added since
        the last query.
        """
        self._ensure_duration_current()
        return self._duration


    # INTERNAL HELPERS ------------------------------------------------------------------

    def _rebuild_local_parameter_container(self) -> None:
        """Ensure local parameter list has one entry per experiment."""
        while len(self.local_parameters) < len(self.experiments):
            self.local_parameters.append([])


    def _update_duration_from_measurements(self) -> None:
        """Update each runner's duration from its measurement time extent.

        Explicit durations passed to :meth:`add_experiment` act as a floor.
        Called lazily via :meth:`_ensure_duration_current`.
        """
        for exp in self.experiments:
            if not exp.measurements:
                continue

            derived = float(max(meas.time.max() for meas in exp.measurements))
            dur = max(float(exp.duration), derived) if exp.duration is not None else derived

            if hasattr(exp.runner, "duration"):
                exp.runner.duration = dur

        all_meas = [m for exp in self.experiments for m in exp.measurements]
        self._duration = float(max((m.time.max() for m in all_meas), default=0.0))
        self._duration_dirty = False


    def _ensure_duration_current(self) -> None:
        """Run duration update only when measurements have changed since last call."""
        if self._duration_dirty:
            self._update_duration_from_measurements()


    def _ensure_experiment(self, idx: int) -> None:
        """Ensure experiment indices up to *idx* exist, cloning the base sim if needed."""
        if idx < 0:
            raise IndexError("experiment index must be >= 0")

        while len(self.experiments) <= idx:
            if self._base_simulator is None:
                raise IndexError(
                    f"experiment index {idx} out of range and no base simulator "
                    "is available for cloning. Call add_experiment() explicitly."
                )
            self.add_experiment(
                self._base_simulator,
                copy_sim=True,
                **self._default_runner_kwargs,
            )

        self._rebuild_local_parameter_container()


    @staticmethod
    def _block_occurrence(sim, block_obj: object) -> tuple[str, int]:
        """Return ``(type_name, occurrence_index)`` for *block_obj* in ``sim.blocks``."""
        tname = type(block_obj).__name__
        occ = 0
        for b in sim.blocks:
            if type(b).__name__ == tname:
                if b is block_obj:
                    return tname, occ
                occ += 1
        raise ValueError("block not found in sim.blocks")


    @staticmethod
    def _find_block_by_occurrence(sim, type_name: str, occurrence: int) -> object:
        """Find the *N*-th block of the given class name in ``sim.blocks``."""
        occ = 0
        for b in sim.blocks:
            if type(b).__name__ == type_name:
                if occ == occurrence:
                    return b
                occ += 1
        raise ValueError(
            f"No block '{type_name}' occurrence {occurrence} found in simulation"
        )


    def _resolve_output(self, exp: Experiment, sig: ScopeSignal) -> ScopeSignal:
        """Resolve a :class:`ScopeSignal` to the correct scope in an experiment sim."""
        if sig.scope is not None:
            return sig

        sim = getattr(exp.runner, "sim", None)
        if sim is None:
            raise ValueError(
                "Cannot resolve ScopeSignal without a SimRunner-backed experiment"
            )

        if sig.block_type is None or sig.occurrence is None:
            raise ValueError(
                "Unresolvable ScopeSignal (missing selector and scope reference)"
            )

        scope_obj = self._find_block_by_occurrence(sim, sig.block_type, sig.occurrence)
        return ScopeSignal(
            scope=scope_obj,
            port=int(sig.port),
            block_type=sig.block_type,
            occurrence=sig.occurrence,
        )


    @staticmethod
    def _scope_port_from_signal(signal: object) -> tuple[object, int]:
        """Extract ``(scope, port)`` from a PortReference or ``(scope, port)`` tuple.

        Parameters
        ----------
        signal : PortReference or tuple
            Either ``scope[0]`` (PortReference with ``.block`` and ``.ports``) or
            a ``(scope, port)`` tuple.

        Returns
        -------
        scope : object
        port : int
        """
        if isinstance(signal, (tuple, list)) and len(signal) == 2:
            return signal[0], int(signal[1])

        blk = getattr(signal, "block", None)
        ports = getattr(signal, "ports", None)
        if blk is not None and ports is not None:
            if len(ports) != 1:
                raise ValueError(
                    f"Expected PortReference with a single port, got {len(ports)}."
                )
            return blk, int(ports[0])

        raise TypeError(
            f"Unsupported signal type {type(signal).__name__}. "
            "Use signal=scope[0] (PortReference) or signal=(scope, port)."
        )


    # CONFIGURATION API -----------------------------------------------------------------

    def add_experiment(
        self,
        simulator,
        *,
        duration: float | None = None,
        adaptive: bool | object = _UNSET,
        pre_run: Callable[[], None] | None | object = _UNSET,
        reset_time: float = 0.0,
        suppress_reset_log: bool = True,
        copy_sim: bool = False,
    ) -> int:
        """Register a new experiment and return its index.

        Parameters
        ----------
        simulator : object
            PathSim simulation or SimRunner-like object.
        duration : float, optional
            Explicit simulation duration. If not set, derived from measurements.
        adaptive : bool, optional
            Enable adaptive (variable) step-size control.  Pass ``True`` when
            the simulation solver is configured with error tolerances
            (``tolerance_lte_rel`` / ``tolerance_lte_abs``).  Pass ``False``
            (default) for fixed-step simulations.  If not provided, inherits
            the value passed to the :class:`ParameterEstimator` constructor.
        pre_run : callable, optional
            Callable executed just before each ``sim.run()``.  If not
            provided, inherits the constructor default.
        reset_time : float
            Time passed to ``sim.reset()``.
        suppress_reset_log : bool
            Suppress sim logging during reset/run.
        copy_sim : bool
            Deep-copy *simulator* so this experiment has independent state.

        Returns
        -------
        int
            Index of the newly registered experiment.
        """
        # Inherit constructor defaults when kwargs are not explicitly provided
        if adaptive is _UNSET:
            adaptive = self._default_adaptive
        if pre_run is _UNSET:
            pre_run = self._default_pre_run
        if hasattr(simulator, "run"):
            if not copy_sim and self.experiments:
                existing_sims = [
                    getattr(exp.runner, "sim", None) for exp in self.experiments
                ]
                if any(s is simulator for s in existing_sims):
                    warnings.warn(
                        "add_experiment() called with copy_sim=False, but this "
                        "simulator is already registered in another experiment. "
                        "Both experiments will share simulation state, which "
                        "produces incorrect results in multi-experiment fitting. "
                        "Pass copy_sim=True to give each experiment independent state.",
                        UserWarning,
                        stacklevel=2,
                    )

            sim_obj = _deepcopy_sim(simulator) if copy_sim else simulator
            runner = SimRunner(
                sim=sim_obj,
                output=None,
                duration=float(duration) if duration is not None else 0.0,
                adaptive=adaptive,
                pre_run=pre_run,
                reset_time=reset_time,
                suppress_reset_log=suppress_reset_log,
            )
        else:
            runner = simulator

        self.experiments.append(
            Experiment(
                runner=runner,
                duration=float(duration) if duration is not None else None,
            )
        )
        self._rebuild_local_parameter_container()
        self._duration_dirty = True
        return len(self.experiments) - 1


    def add_timeseries(
        self,
        ts: TimeSeriesData,
        *,
        scope: object | None = None,
        port: int | None = None,
        signal: object | None = None,
        sigma: float | None = None,
        experiment: int = 0,
    ) -> "ParameterEstimator":
        """Add a measurement dataset and map it to a scope output for an experiment.

        Parameters
        ----------
        ts : TimeSeriesData
            Measurement data.
        scope : object, optional
            Scope block with ``.read()``.
        port : int, optional
            Port index on *scope*.
        signal : object, optional
            PortReference (``scope[0]``). Pass either ``signal=`` or
            ``scope=`` / ``port=``, not both.
        sigma : float, optional
            Measurement noise standard deviation used to normalize residuals:
            ``r = (y_pred - y_meas) / sigma``.  A larger value down-weights
            noisy measurements relative to others.  Defaults to 1.0.
        experiment : int
            Experiment index to attach this dataset to.

        Returns
        -------
        ParameterEstimator
            Self, for method chaining.
        """
        if not isinstance(ts, TimeSeriesData):
            raise TypeError(
                f"add_timeseries expects TimeSeriesData, got {type(ts).__name__}"
            )

        self._ensure_experiment(experiment)

        if signal is not None:
            if scope is not None or port is not None:
                raise ValueError(
                    "Pass either `signal=` OR (`scope=`, `port=`), not both."
                )
            scope, port = self._scope_port_from_signal(signal)

        if scope is None or port is None:
            raise ValueError(
                "You must pass either scope=..., port=... or signal=scope[0]."
            )

        # Store (type, occurrence) so the correct deep-copied scope can be
        # resolved at run time for each experiment.
        block_type = None
        occurrence = None
        sim0 = getattr(self.experiments[0].runner, "sim", None) if self.experiments else None
        if sim0 is not None:
            try:
                block_type, occurrence = self._block_occurrence(sim0, scope)
            except Exception:
                pass

        exp = self.experiments[experiment]
        exp.measurements.append(ts)
        exp.outputs.append(
            ScopeSignal(
                scope=None if block_type is not None else scope,
                port=int(port),
                block_type=block_type,
                occurrence=occurrence,
            )
        )
        exp.sigma.append(float(sigma) if sigma is not None else None)

        self._duration_dirty = True
        self._cached_x = None          # measurement change invalidates residuals cache
        self._cached_residuals = None
        return self


    def add_block_parameter(
        self,
        block,
        param_name: str,
        value: float | None = None,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        param_id: str | None = None,
        transform: Callable[[float], float] | None = None,
    ) -> "ParameterEstimator":
        """Add a block attribute as a fitted parameter (single-experiment).

        This is the recommended way to register a block attribute for fitting
        in single-experiment problems.  The optimizer reads and writes the
        attribute on *block* directly during each evaluation.

        For **multi-experiment** problems where the same attribute should be
        shared across deep-copied simulations, use
        :meth:`add_global_block_parameter` instead.

        For **free parameters** that live in Python closures (not bound to any
        block attribute), use :meth:`add_parameters` with a
        :class:`Parameter` object instead.

        Parameters
        ----------
        block : object
            Target block instance.
        param_name : str
            Attribute name on the block (e.g. ``"gain"``).
        value : float, optional
            Initial optimizer-space value; defaults to the block's current
            attribute value.
        bounds : tuple[float, float]
            Lower / upper bounds in optimizer space.
        param_id : str, optional
            Human-readable prefix for the parameter name used in display output.
        transform : callable, optional
            Mapping from optimizer space to model space (e.g. ``np.exp`` to
            enforce positivity, ``scipy.special.exp10`` for log10 space).

        Returns
        -------
        ParameterEstimator
        """
        self.global_parameters.append(
            block_param_to_var(
                block,
                param_name,
                value=value,
                bounds=bounds,
                param_id=param_id,
                transform=transform,
            )
        )
        self._invalidate_cache()
        return self


    def add_parameters(self, params: list[Parameter]) -> "ParameterEstimator":
        """Add pre-constructed :class:`Parameter` objects as global parameters.

        Use this method when parameters are **free** — i.e. they live in a
        Python closure and are read directly by the model code via ``p()``
        rather than being mapped to a block attribute::

            k = Parameter("k", value=1.0, bounds=(0, 5), transform=np.exp)

            def ode_rhs(x, u, t):
                return -k() * x          # k() returns the model-space value

            est = ParameterEstimator(simulator=sim, parameters=[k])
            est.add_parameters([k])      # or pass parameters= to constructor

        For block-bound parameters (writing to a block attribute), prefer
        :meth:`add_block_parameter` — it is simpler and does not require
        constructing a :class:`Parameter` manually.

        Parameters
        ----------
        params : list[Parameter]
            :class:`Parameter` instances to register as global parameters.

        Returns
        -------
        ParameterEstimator
        """
        self.global_parameters.extend(params)
        self._invalidate_cache()
        return self


    def add_global_block_parameter(
        self,
        block_name: str,
        param_name: str,
        *,
        value: float | None = None,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        param_id: str | None = None,
        transform: Callable[[float], float] | None = None,
    ) -> "ParameterEstimator":
        """Add a global block parameter shared across all experiments.

        Locates the first block of type *block_name* in each experiment sim
        and creates a :class:`SharedBlockParameter` that applies the same
        optimizer value to all of them simultaneously.

        Parameters
        ----------
        block_name : str
            Block class name to match (e.g. ``"Constant"``).
        param_name : str
            Attribute name on the block.
        value : float, optional
            Initial value; defaults to the attribute's current value in
            experiment 0.
        bounds : tuple[float, float]
            Lower / upper bounds.
        param_id : str, optional
            Human-readable prefix for the parameter name.
        transform : callable, optional
            Optimizer-to-model transform.

        Returns
        -------
        ParameterEstimator
        """
        if not self.experiments:
            raise ValueError(
                "No experiments configured. Pass simulator= or call add_experiment()."
            )

        targets: list[Any] = []
        for exp in self.experiments:
            sim = getattr(exp.runner, "sim", None)
            if sim is None:
                raise ValueError(
                    "Global block parameters require SimRunner experiments."
                )
            match = next(
                (b for b in sim.blocks if type(b).__name__ == block_name), None
            )
            if match is None:
                raise ValueError(
                    f"Experiment sim has no block of type '{block_name}'."
                )
            if not hasattr(match, param_name):
                raise AttributeError(
                    f"Block '{block_name}' has no attribute '{param_name}'"
                )
            targets.append(match)

        pname = (
            f"{param_id}.{param_name}"
            if param_id is not None
            else f"{block_name}.{param_name}"
        )
        if value is None:
            value = float(getattr(targets[0], param_name))

        self.global_parameters.append(
            SharedBlockParameter(
                name=pname,
                targets=targets,
                attribute=param_name,
                value=float(value),
                bounds=bounds,
                transform=transform,
            )
        )
        self._invalidate_cache()
        return self


    def add_local_block_parameter(
        self,
        experiment: int,
        block_name: str,
        param_name: str,
        *,
        value: float | None = None,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        param_id: str | None = None,
        transform: Callable[[float], float] | None = None,
    ) -> "ParameterEstimator":
        """Add an experiment-local block parameter (distinct value per experiment).

        Parameters
        ----------
        experiment : int
            Experiment index.
        block_name : str
            Block class name to match.
        param_name : str
            Attribute name on the block.
        value : float, optional
            Initial value; defaults to the block's current attribute value.
        bounds : tuple[float, float]
            Lower / upper bounds.
        param_id : str, optional
            Human-readable prefix for the parameter name.
        transform : callable, optional
            Optimizer-to-model transform.

        Returns
        -------
        ParameterEstimator
        """
        self._ensure_experiment(experiment)

        exp = self.experiments[experiment]
        sim = getattr(exp.runner, "sim", None)
        if sim is None:
            raise ValueError("Local block parameters require SimRunner experiments.")

        match = next(
            (b for b in sim.blocks if type(b).__name__ == block_name), None
        )
        if match is None:
            raise ValueError(
                f"Experiment {experiment} sim has no block of type '{block_name}'."
            )
        if not hasattr(match, param_name):
            raise AttributeError(
                f"Block '{block_name}' has no attribute '{param_name}'"
            )

        pname = (
            f"{param_id}.exp{experiment}.{param_name}"
            if param_id is not None
            else f"exp{experiment}.{block_name}.{param_name}"
        )
        if value is None:
            value = float(getattr(match, param_name))

        self.local_parameters[experiment].append(
            BlockParameter(
                block=match,
                attribute=param_name,
                name=pname,
                value=float(value),
                bounds=bounds,
                transform=transform,
            )
        )
        self._invalidate_cache()
        return self


    # OPTIMIZATION ENGINE ---------------------------------------------------------------

    def _validate_fit_inputs(self) -> None:
        """Raise early with clear messages for degenerate fit configurations."""
        self._ensure_duration_current()

        if not self.experiments:
            raise ValueError(
                "No experiments configured. "
                "Pass simulator= to the constructor or call add_experiment()."
            )

        if not self.parameters:
            raise ValueError(
                "No parameters to fit. Add parameters with add_block_parameter(), "
                "add_parameters(), or add_global_block_parameter()."
            )

        if not any(exp.measurements for exp in self.experiments):
            raise ValueError(
                "No measurements provided. "
                "Add measurement data with add_timeseries()."
            )

        # Validate bounds consistency
        for p in self.parameters:
            lo, hi = p.bounds
            if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
                raise ValueError(
                    f"Parameter '{p.name}': lower bound {lo} > upper bound {hi}"
                )


    def apply(self, x: np.ndarray) -> None:
        """Apply the optimizer parameter vector to all parameters.

        Vector order: global parameters first, then locals experiment-by-experiment.

        Parameters
        ----------
        x : array_like
            Optimizer-space parameter vector of length ``len(self.parameters)``.
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        expected = len(self.parameters)
        if x_arr.size != expected:
            raise ValueError(f"Expected x of length {expected}, got {x_arr.size}")

        k = 0
        for p in self.global_parameters:
            p.set(float(x_arr[k]))
            k += 1

        for exp_params in self.local_parameters:
            for p in exp_params:
                p.set(float(x_arr[k]))
                k += 1


    def simulate(
        self,
        x: np.ndarray,
        output_idx: int = 0,
        *,
        experiment: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a single experiment and return one of its mapped outputs.

        Parameters
        ----------
        x : array_like
            Parameter vector passed to :meth:`apply`.
        output_idx : int
            Index of the output to return among the datasets registered for
            this experiment.
        experiment : int
            Experiment index.

        Returns
        -------
        t : np.ndarray
        y : np.ndarray
        """
        if not self.experiments:
            raise ValueError("No experiments configured.")

        if experiment < 0 or experiment >= len(self.experiments):
            raise IndexError(
                f"experiment index {experiment} out of range "
                f"(0..{len(self.experiments) - 1})"
            )

        self._ensure_duration_current()
        self.apply(x)

        exp = self.experiments[experiment]
        exp.runner.run()

        if output_idx < 0 or output_idx >= len(exp.outputs):
            raise IndexError(
                f"output_idx {output_idx} out of range for experiment {experiment}"
            )

        return self._resolve_output(exp, exp.outputs[output_idx]).read()


    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute the stacked residual vector across all experiments and datasets.

        Parameters
        ----------
        x : array_like
            Optimizer-space parameter vector.

        Returns
        -------
        np.ndarray
            Concatenated ``(y_pred - y_meas) / sigma`` arrays, one block per
            dataset per experiment.
        """
        if not self.experiments:
            raise ValueError("No experiments configured.")

        self._ensure_duration_current()
        x_arr = np.asarray(x, dtype=float).reshape(-1)

        # Return cached result if x is unchanged
        if (
            self._cached_x is not None
            and self._cached_residuals is not None
            and x_arr.shape == self._cached_x.shape
            and np.array_equal(x_arr, self._cached_x)
        ):
            return self._cached_residuals.copy()

        self.apply(x_arr)

        all_residuals: list[np.ndarray] = []

        for exp_idx, exp in enumerate(self.experiments):
            if not exp.measurements:
                continue

            if len(exp.outputs) != len(exp.measurements):
                raise ValueError(
                    f"Experiment {exp_idx} has {len(exp.measurements)} "
                    f"measurement(s) but {len(exp.outputs)} output mapping(s)."
                )

            exp.runner.run()

            for i, meas in enumerate(exp.measurements):
                out = self._resolve_output(exp, exp.outputs[i])
                t_out, y_out = out.read()

                t_out = np.asarray(t_out, dtype=float).reshape(-1)
                y_out = np.asarray(y_out, dtype=float).reshape(-1)

                y_pred = np.interp(meas.time, t_out, y_out)
                meas_data = np.asarray(meas.data, dtype=float).reshape(-1)

                sigma_i = exp.sigma[i] if exp.sigma[i] is not None else 1.0
                all_residuals.append((y_pred - meas_data) / sigma_i)

        result = np.concatenate(all_residuals) if all_residuals else np.array([], dtype=float)

        self._cached_x = x_arr.copy()
        self._cached_residuals = result.copy()
        return result


    def fit(
        self,
        *,
        x0: Sequence[float] | None = None,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        loss: str = "linear",
        f_scale: float = 1.0,
        max_nfev: int = 80,
        verbose: int = 0,
        method: str = "least_squares",
        constraints: list[dict] | None = None,
    ) -> EstimatorResult:
        """Fit parameters to measurements using SciPy optimizers.

        Parameters
        ----------
        x0 : sequence of float, optional
            Initial optimizer-space parameter vector; extracted from parameter
            values by default.
        bounds : (lower, upper), optional
            Bounds in optimizer space; extracted from parameters by default.
        loss : str
            Robust loss function for ``scipy.optimize.least_squares``
            (``"linear"``, ``"soft_l1"``, ``"huber"``, ``"cauchy"``,
            ``"arctan"``).  **Only used when** ``method="least_squares"``;
            ignored for all ``scipy.optimize.minimize`` methods.
            Use ``"linear"`` (default) when measurements have Gaussian noise.
            Use ``"soft_l1"`` or ``"huber"`` when the data contains outliers.
        f_scale : float
            Residual scale for robust loss functions (``least_squares`` only).
            Residuals smaller than ``f_scale`` are treated as inliers and
            minimised normally; residuals larger than ``f_scale`` are
            downweighted by the chosen ``loss`` function.  Set this to the
            expected magnitude of a *good* residual — typically the
            measurement noise level.  For example, if measurements are in
            units of milliamps and noise is roughly ±2 mA, use
            ``f_scale=2``.  Has no effect when ``loss="linear"``.
        max_nfev : int
            Maximum function evaluations (``least_squares``) or iterations
            (``minimize``).
        verbose : int
            Verbosity: 0 = silent, 1 = final summary, 2 = per-iteration
            (``least_squares`` only; ``minimize`` uses scipy's own display).
        method : str
            ``"least_squares"`` (default) or any ``scipy.optimize.minimize``
            method such as ``"L-BFGS-B"``, ``"SLSQP"``, or ``"trust-constr"``.
        constraints : list of dict, optional
            Constraint definitions for ``scipy.optimize.minimize``.
            Only supported with ``"SLSQP"``, ``"trust-constr"``, or
            ``"COBYLA"``.

        Returns
        -------
        EstimatorResult

        Notes
        -----
        ``"least_squares"`` is generally preferred: it exploits the residual
        structure, supports robust loss functions, and converges faster on
        well-conditioned problems.  Use ``minimize`` methods only when
        constraints are required or the problem does not fit the least-squares
        framework.
        """
        self._validate_fit_inputs()

        if x0 is None:
            x0 = [p.value for p in self.parameters]
        x0_arr = np.asarray(x0, dtype=float)

        if bounds is None:
            lower = np.array([p.bounds[0] for p in self.parameters], dtype=float)
            upper = np.array([p.bounds[1] for p in self.parameters], dtype=float)
            bounds_arr = (lower, upper)
        else:
            bounds_arr = (
                np.asarray(bounds[0], dtype=float),
                np.asarray(bounds[1], dtype=float),
            )

        _CONSTRAINT_METHODS = {"SLSQP", "trust-constr", "COBYLA"}

        if constraints is not None and method not in _CONSTRAINT_METHODS:
            raise ValueError(
                f"Method '{method}' does not support general constraints. "
                "Supported constraint methods: 'SLSQP', 'trust-constr', 'COBYLA'."
            )

        # ── least_squares path ────────────────────────────────────────────────
        if method == "least_squares":
            res = sci_opt.least_squares(
                self.residuals,
                x0=x0_arr,
                bounds=bounds_arr,
                loss=loss,
                f_scale=float(f_scale),
                max_nfev=int(max_nfev),
                verbose=int(verbose),
            )

            return EstimatorResult(
                x=res.x,
                cost=float(res.cost),
                nfev=int(res.nfev),
                success=bool(res.success),
                message=str(res.message),
            )

        # ── scipy.optimize.minimize path ─────────────────────────────────────
        bounds_list = list(zip(bounds_arr[0], bounds_arr[1]))

        def objective(xk: np.ndarray) -> float:
            r = self.residuals(xk)
            return float(0.5 * np.dot(r, r))

        # Build solver options.  'disp' is deprecated for some methods (e.g.
        # L-BFGS-B) in recent SciPy; skip it when the method does not support
        # it, and rely on the caller's own logging instead.
        _DISP_METHODS = {"SLSQP", "trust-constr", "COBYLA", "Nelder-Mead", "Powell"}
        opts: dict = {"maxiter": int(max_nfev)}
        if verbose > 0 and method in _DISP_METHODS:
            opts["disp"] = True

        res = sci_opt.minimize(
            objective,
            x0=x0_arr,
            bounds=bounds_list,
            method=method,
            constraints=constraints,
            options=opts,
        )

        return EstimatorResult(
            x=res.x,
            cost=float(res.fun),
            nfev=int(getattr(res, "nfev", max_nfev)),
            success=bool(res.success),
            message=str(res.message),
        )


    # NESTED SCHUR OPTIMIZATION ---------------------------------------------------------

    def fit_nested(
        self,
        *,
        x0_G: "Sequence[float] | np.ndarray | None" = None,
        max_outer_nfev: int = 50,
        max_inner_nfev: int = 30,
        eps: float | None = None,
        loss: str = "linear",
        f_scale: float = 1.0,
        outer_method: str = "least_squares",
        outer_constraints: "list[dict] | None" = None,
        verbose: int = 0,
    ) -> "EstimatorResult":
        """Bilevel nested Schur optimization for multi-experiment problems.

        Outer loop (global parameters)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Drives Gauss-Newton updates on the global parameters ``θ_G`` using the
        *reduced* Jacobian and the Schur complement as the reduced Hessian::

            Δθ_G = −S_G⁻¹ g_G_red

        where

        * ``S_G = Σᵢ J_red_iᵀ J_red_i`` — Schur complement (reduced Hessian)
        * ``g_G_red = Σᵢ J_red_iᵀ r_i*`` — reduced gradient

        The reduced Jacobian for experiment *i* is obtained by projecting out
        the directions explained by the local parameters::

            J_red_i = P_{L,i} J_{G,i}

        where ``P_{L,i} = I − J_{L,i}(J_{L,i}ᵀJ_{L,i})⁻¹J_{L,i}ᵀ`` is the
        orthogonal projector onto the complement of the column space of
        ``J_{L,i}``.  Computed stably as::

            A = lstsq(J_Li, [r_i | J_Gi])
            r_red_i = r_i − J_Li @ A[:, 0]     # P_{L,i} r_i
            J_red_i = J_Gi − J_Li @ A[:, 1:]   # P_{L,i} J_{G,i}

        Inner loop (local parameters, one per experiment)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        For each fixed ``θ_G``, every experiment's local parameters
        ``θ_{L,i}`` are optimised independently via
        ``scipy.optimize.least_squares``.  The inner problems are fully
        decoupled — each only involves one experiment's data.

        Parameters
        ----------
        x0_G : array_like, optional
            Initial global parameter vector in optimizer space.  Defaults to
            the current global parameter values.
        max_outer_nfev : int
            Maximum outer function evaluations.  Each counts as one call to
            the reduced residual/Jacobian, which internally solves all inner
            problems plus ``n_G`` finite-difference perturbations per
            experiment.
        max_inner_nfev : int
            Maximum function evaluations for each inner
            ``scipy.optimize.least_squares`` call.
        eps : float, optional
            Relative finite-difference step for Jacobian computation.
            Defaults to ``√(machine epsilon) ≈ 1.5e-8``.
        loss : str
            Robust loss for the outer problem (``"linear"``, ``"soft_l1"``,
            ``"huber"``, …).  **Only used when** ``outer_method="least_squares"``.
        f_scale : float
            Residual scale for robust outer loss.
            **Only used when** ``outer_method="least_squares"``.
        outer_method : str
            Outer optimizer.  ``"least_squares"`` (default) uses
            ``scipy.optimize.least_squares`` (trust-region reflective) and
            supports ``loss`` / ``f_scale``.  Any
            ``scipy.optimize.minimize`` method string (e.g.
            ``"trust-constr"``, ``"trust-ncg"``, ``"L-BFGS-B"``) switches
            to the minimize path, where the outer objective is
            ``½‖r_red‖²`` with gradient ``J_red^T r_red`` and (for
            Hessian-based methods) the Gauss-Newton approximation
            ``J_red^T J_red``.  Use ``"trust-constr"`` when nonlinear
            constraints on the global parameters are required.
        outer_constraints : list of dict, optional
            Constraint definitions passed directly to
            ``scipy.optimize.minimize``.  Only supported for
            ``"SLSQP"``, ``"trust-constr"``, and ``"COBYLA"``.  Ignored
            when ``outer_method="least_squares"``.
        verbose : int
            0 = silent, 1 = print outer cost each iteration, 2 = full scipy
            output.

        Returns
        -------
        EstimatorResult
            ``x`` contains the full parameter vector
            ``[x_G*, x_L0*, x_L1*, …]`` in the same order as
            :attr:`parameters`.  Cached so that :meth:`sensitivity` works
            immediately after without arguments.

        Raises
        ------
        ValueError
            If no global parameters are registered.

        Notes
        -----
        Falls back to :meth:`fit` when no local parameters are present.
        """
        self._validate_fit_inputs()

        n_G = len(self.global_parameters)
        if n_G == 0:
            raise ValueError(
                "fit_nested requires at least one global parameter. "
                "Use fit() for single-experiment problems."
            )

        _OUTER_CONSTRAINT_METHODS = {"SLSQP", "trust-constr", "COBYLA"}
        if (
            outer_constraints is not None
            and outer_method != "least_squares"
            and outer_method not in _OUTER_CONSTRAINT_METHODS
        ):
            raise ValueError(
                f"outer_method='{outer_method}' does not support constraints. "
                "Supported constraint methods: 'SLSQP', 'trust-constr', 'COBYLA'."
            )

        has_locals = any(len(lp) > 0 for lp in self.local_parameters)
        if not has_locals:
            x0_list = (
                [p.value for p in self.global_parameters]
                if x0_G is None else list(x0_G)
            )
            return self.fit(
                x0=x0_list, loss=loss, f_scale=f_scale,
                max_nfev=max_outer_nfev, verbose=verbose,
            )

        self._ensure_duration_current()
        rel = eps if eps is not None else np.sqrt(np.finfo(float).eps)

        # ── Initial vectors and bounds ───────────────────────────────────────
        x_G0 = (
            np.array([p.value for p in self.global_parameters], dtype=float)
            if x0_G is None else np.asarray(x0_G, dtype=float).reshape(-1)
        )
        x_G_lo = np.array([p.bounds[0] for p in self.global_parameters], dtype=float)
        x_G_hi = np.array([p.bounds[1] for p in self.global_parameters], dtype=float)

        # Per-experiment local vectors — warm-started from current values
        n_exp = len(self.experiments)
        x_Ls     = [np.array([p.value for p in self.local_parameters[i]], dtype=float)
                    for i in range(n_exp)]
        x_Ls_lo  = [np.array([p.bounds[0] for p in self.local_parameters[i]], dtype=float)
                    for i in range(n_exp)]
        x_Ls_hi  = [np.array([p.bounds[1] for p in self.local_parameters[i]], dtype=float)
                    for i in range(n_exp)]

        # ── Per-experiment residual function ─────────────────────────────────
        def _resid_exp(exp_idx: int, x_G: np.ndarray, x_Li: np.ndarray) -> np.ndarray:
            """Run one experiment and return its weighted residual vector."""
            for j, p in enumerate(self.global_parameters):
                p.set(float(x_G[j]))
            for j, p in enumerate(self.local_parameters[exp_idx]):
                p.set(float(x_Li[j]))
            exp = self.experiments[exp_idx]
            exp.runner.run()
            parts: list[np.ndarray] = []
            for k, meas in enumerate(exp.measurements):
                out    = self._resolve_output(exp, exp.outputs[k])
                t_o, y_o = out.read()
                y_pred = np.interp(
                    meas.time,
                    np.asarray(t_o, dtype=float),
                    np.asarray(y_o, dtype=float),
                )
                sig = exp.sigma[k] if exp.sigma[k] is not None else 1.0
                parts.append(
                    (y_pred - np.asarray(meas.data, dtype=float).reshape(-1)) / sig
                )
            return np.concatenate(parts) if parts else np.array([], dtype=float)

        # ── Outer: build reduced residuals + reduced Jacobian ────────────────
        def _outer(x_G: np.ndarray):
            """
            For each experiment i:
              1. Solve inner problem  →  x_Li*, r_i*
              2. Compute J_Gi  (FD, locals frozen at x_Li*)
              3. Compute J_Li  (FD, x_G fixed)
              4. Reduce:  solve J_Li @ A = [r_i | J_Gi]
                          r_red_i = r_i - J_Li @ A[:,0]
                          J_red_i = J_Gi - J_Li @ A[:,1:]
            Returns (r_red_stacked, J_red_stacked).
            """
            all_r: list[np.ndarray] = []
            all_J: list[np.ndarray] = []

            for exp_idx, exp in enumerate(self.experiments):
                if not exp.measurements:
                    continue

                n_Li = (
                    len(self.local_parameters[exp_idx])
                    if exp_idx < len(self.local_parameters) else 0
                )

                # ── Inner: solve local parameters ─────────────────────────
                if n_Li > 0:
                    inner = sci_opt.least_squares(
                        lambda xl, ei=exp_idx: _resid_exp(ei, x_G, xl),
                        x0=x_Ls[exp_idx],
                        bounds=(x_Ls_lo[exp_idx], x_Ls_hi[exp_idx]),
                        max_nfev=max_inner_nfev,
                    )
                    x_Ls[exp_idx] = inner.x
                    r_i = inner.fun
                else:
                    r_i = _resid_exp(exp_idx, x_G, np.array([]))

                n_r = len(r_i)

                # ── J_Gi: FD w.r.t. globals (locals frozen at x_Li*) ──────
                J_Gi = np.empty((n_r, n_G))
                for j in range(n_G):
                    h = rel * max(1.0, abs(x_G[j]))
                    xp = x_G.copy(); xp[j] += h
                    J_Gi[:, j] = (_resid_exp(exp_idx, xp, x_Ls[exp_idx]) - r_i) / h

                if n_Li > 0:
                    # ── J_Li: FD w.r.t. locals (x_G fixed) ───────────────
                    J_Li = np.empty((n_r, n_Li))
                    for j in range(n_Li):
                        h = rel * max(1.0, abs(x_Ls[exp_idx][j]))
                        xl = x_Ls[exp_idx].copy(); xl[j] += h
                        J_Li[:, j] = (_resid_exp(exp_idx, x_G, xl) - r_i) / h

                    # ── Reduce via orthogonal projection ──────────────────
                    # Solve J_Li @ A = [r_i | J_Gi]  (one lstsq for both)
                    rhs = np.column_stack([r_i.reshape(-1, 1), J_Gi])
                    A, _, _, _ = np.linalg.lstsq(J_Li, rhs, rcond=None)
                    r_red_i = r_i    - J_Li @ A[:, 0]
                    J_red_i = J_Gi   - J_Li @ A[:, 1:]
                else:
                    r_red_i = r_i
                    J_red_i = J_Gi

                all_r.append(r_red_i)
                all_J.append(J_red_i)

            return np.concatenate(all_r), np.vstack(all_J)

        # ── Cache: scipy calls fun then jac with the same x in each step ────
        _cache: dict = {"x": None, "r": None, "J": None}

        def _ensure(x_G: np.ndarray) -> None:
            if _cache["x"] is None or not np.array_equal(x_G, _cache["x"]):
                _cache["r"], _cache["J"] = _outer(x_G)
                _cache["x"] = x_G.copy()
                if verbose >= 1:
                    cost = float(0.5 * np.dot(_cache["r"], _cache["r"]))
                    print(f"  [nested] outer cost = {cost:.6g}")

        def _outer_fun(x_G: np.ndarray) -> np.ndarray:
            _ensure(x_G); return _cache["r"]

        def _outer_jac(x_G: np.ndarray) -> np.ndarray:
            _ensure(x_G); return _cache["J"]

        # ── Run outer optimisation ───────────────────────────────────────────
        if outer_method == "least_squares":
            outer_res = sci_opt.least_squares(
                _outer_fun,
                x0=x_G0,
                jac=_outer_jac,
                bounds=(x_G_lo, x_G_hi),
                loss=loss,
                f_scale=float(f_scale),
                max_nfev=max_outer_nfev,
                verbose=max(0, verbose - 1),
            )
            x_G_opt       = outer_res.x
            outer_cost    = float(outer_res.cost)
            outer_nfev    = int(outer_res.nfev)
            outer_success = bool(outer_res.success)
            outer_message = str(outer_res.message)

        else:
            # ── minimize path: scalar ½‖r_red‖² with GN gradient/Hessian ──
            def _scalar_obj(x_G: np.ndarray) -> float:
                r = _outer_fun(x_G)
                return float(0.5 * np.dot(r, r))

            def _scalar_grad(x_G: np.ndarray) -> np.ndarray:
                r = _outer_fun(x_G)
                J = _outer_jac(x_G)
                return J.T @ r

            def _gn_hess(x_G: np.ndarray) -> np.ndarray:
                J = _outer_jac(x_G)
                return J.T @ J

            _HESS_METHODS   = {"trust-ncg", "trust-exact", "trust-krylov"}
            _DISP_METHODS   = {"SLSQP", "trust-constr", "COBYLA",
                                "Nelder-Mead", "Powell"}

            opts: dict = {"maxiter": int(max_outer_nfev)}
            if verbose > 0 and outer_method in _DISP_METHODS:
                opts["disp"] = True

            minimize_kwargs: dict = dict(
                fun=_scalar_obj,
                x0=x_G0,
                jac=_scalar_grad,
                method=outer_method,
                bounds=list(zip(x_G_lo, x_G_hi)),
                constraints=outer_constraints,
                options=opts,
            )
            if outer_method in _HESS_METHODS:
                minimize_kwargs["hess"] = _gn_hess

            outer_res     = sci_opt.minimize(**minimize_kwargs)
            x_G_opt       = outer_res.x
            r_final       = _outer_fun(x_G_opt)
            outer_cost    = float(0.5 * np.dot(r_final, r_final))
            outer_nfev    = int(getattr(outer_res, "nfev", max_outer_nfev))
            outer_success = bool(outer_res.success)
            outer_message = str(outer_res.message)

        # Ensure x_Ls is consistent with the converged x_G_opt
        _ensure(x_G_opt)

        # ── Store full solution ──────────────────────────────────────────────
        x_full = np.concatenate([x_G_opt] + x_Ls)
        self.apply(x_full)
        self._cached_x        = x_full.copy()
        self._cached_residuals = None

        return EstimatorResult(
            x=x_full,
            cost=outer_cost,
            nfev=outer_nfev,
            success=outer_success,
            message=outer_message,
        )


    # SENSITIVITY & IDENTIFIABILITY -----------------------------------------------------

    def sensitivity(
        self,
        x: "Sequence[float] | np.ndarray | None" = None,
        *,
        eps: float | None = None,
    ) -> "SensitivityResult":
        """Compute local sensitivity and practical identifiability at ``x``.

        Evaluates the weighted Jacobian **J** = ``∂r/∂θ`` by finite
        differences, then derives the Fisher Information Matrix and the
        parameter covariance to assess which parameters are well-constrained
        and which are correlated or unidentifiable.

        Parameters
        ----------
        x : array_like, optional
            Parameter vector (optimizer space) at which to evaluate.
            Defaults to the vector cached from the most recent
            :meth:`fit` call so that ``est.sensitivity()`` works right
            after fitting.
        eps : float, optional
            Relative finite-difference step size.  Defaults to
            ``√(machine epsilon) ≈ 1.5e-8``.  Increase for noisy or
            discontinuous residuals (e.g. ``eps=1e-4``).

        Returns
        -------
        SensitivityResult
            Contains the Jacobian, FIM, covariance, standard errors,
            correlation matrix, eigenvalues, and condition number.
            Call ``.display()`` for a formatted summary or ``.plot()``
            for visualizations.

        Raises
        ------
        ValueError
            If no ``x`` is provided and no previous fit result is cached.

        Notes
        -----
        Each parameter requires one additional simulation run (2-point
        finite differences), so the total cost is ``n_params`` runs.
        The residuals are already normalised by ``sigma`` inside
        :meth:`residuals`, so the resulting standard errors reflect the
        per-measurement noise level without further scaling.

        Examples
        --------
        >>> result = est.fit()
        >>> sens = est.sensitivity()
        >>> sens.display()
        >>> fig, axes = sens.plot()
        """
        from .sensitivity import SchurResult, SensitivityResult

        self._validate_fit_inputs()

        if x is None:
            if self._cached_x is None:
                raise ValueError(
                    "No x provided and no cached fit result available. "
                    "Run fit() first or pass x explicitly."
                )
            x_arr = self._cached_x.copy()
        else:
            x_arr = np.asarray(x, dtype=float).reshape(-1)

        # Finite-difference Jacobian via 2-point forward differences.
        # step_j = rel_step * max(1, |x_j|) to scale with parameter magnitude.
        rel = eps if eps is not None else np.sqrt(np.finfo(float).eps)
        r0  = self.residuals(x_arr)
        jac = np.empty((len(r0), len(x_arr)))
        for j, xj in enumerate(x_arr):
            h         = rel * max(1.0, abs(xj))
            xp        = x_arr.copy()
            xp[j]    += h
            jac[:, j] = (self.residuals(xp) - r0) / h

        # Build model-space values and names from flattened parameter list
        params = self.parameters
        names  = [p.name for p in params]
        values = np.array([
            p.transform(x_arr[i]) if p.transform is not None else x_arr[i]
            for i, p in enumerate(params)
        ])

        # ── Nested Schur complement (multi-experiment, global + local params) ─────
        #
        # When both global and local parameters exist, marginalise out local
        # uncertainty to obtain the effective FIM for the global parameters:
        #
        #   S_G = F_GG  −  Σᵢ  F_{GL_i} · pinv(F_{L_iL_i}) · F_{GL_i}ᵀ
        #
        # Column partition of jac:  [ G (n_G) | L_0 (n_L0) | L_1 (n_L1) | … ]
        # Row    partition of jac:  experiments with measurements, in order.
        #
        n_G = len(self.global_parameters)
        has_locals = any(len(lp) > 0 for lp in self.local_parameters)

        schur: "SchurResult | None" = None

        if n_G > 0 and has_locals:
            F_GG = jac[:, :n_G].T @ jac[:, :n_G]   # global–global FIM block
            S_G  = F_GG.copy()

            row_offset = 0
            col_offset = n_G

            for exp_idx, exp in enumerate(self.experiments):
                n_r_i = (
                    sum(len(m.time) for m in exp.measurements)
                    if exp.measurements else 0
                )
                n_L_i = (
                    len(self.local_parameters[exp_idx])
                    if exp_idx < len(self.local_parameters) else 0
                )

                if n_L_i > 0 and n_r_i > 0:
                    J_Gi   = jac[row_offset:row_offset + n_r_i, :n_G]
                    J_Li   = jac[row_offset:row_offset + n_r_i,
                                 col_offset:col_offset + n_L_i]
                    F_LiLi = J_Li.T @ J_Li
                    F_GLi  = J_Gi.T @ J_Li
                    S_G   -= F_GLi @ np.linalg.pinv(F_LiLi) @ F_GLi.T

                row_offset += n_r_i
                col_offset += n_L_i

            global_values = np.array([
                p.transform(x_arr[i]) if p.transform is not None else x_arr[i]
                for i, p in enumerate(self.global_parameters)
            ])

            schur = SchurResult(
                schur_fim=S_G,
                param_names=[p.name for p in self.global_parameters],
                param_values=global_values,
            )

        return SensitivityResult(
            jacobian=jac,
            param_names=names,
            param_values=values,
            schur=schur,
        )


    # RESULTS AND VISUALIZATION ---------------------------------------------------------

    def display(self) -> None:
        """Print a summary table of all parameters and their current values."""
        print("=" * 60)
        print("Parameter Estimation Results")
        print("=" * 60)

        all_params = list(self.global_parameters)
        for lp in self.local_parameters:
            all_params.extend(lp)
        col_w = max((len(p.name) for p in all_params), default=20) + 2

        def _fmt(p: Parameter) -> None:
            val = p()
            lo, hi = p.bounds
            lo_s = f"{lo:.4g}" if lo != -np.inf else "-inf"
            hi_s = f"{hi:.4g}" if hi != np.inf else "inf"
            bounds_s = (
                f"  [{lo_s}, {hi_s}]"
                if lo != -np.inf or hi != np.inf
                else ""
            )
            if p.transform is not None:
                print(f"  {p.name:{col_w}s}  x={p.value:.6g}  ->  {val:.6g}{bounds_s}")
            else:
                print(f"  {p.name:{col_w}s}  = {val:.6g}{bounds_s}")

        if self.global_parameters:
            print("\nGlobal parameters:")
            print("-" * 40)
            for p in self.global_parameters:
                _fmt(p)

        for i, exp_params in enumerate(self.local_parameters):
            if exp_params:
                print(f"\nLocal parameters (experiment {i}):")
                print("-" * 40)
                for p in exp_params:
                    _fmt(p)

        print("=" * 60)


    def plot_fit(
        self,
        x: np.ndarray,
        *,
        experiments: list[int] | None = None,
        overlay: bool = False,
        show_measurements: bool = True,
        show_predictions: bool = True,
        prediction_style: str | None = None,
        measurement_style: str | None = None,
        fig=None,
        axes=None,
        title: str | None = None,
        xlabel: str = "Time",
        ylabel: str = "Output",
        grid: bool = True,
        legend: bool = True,
    ):
        """Plot measurements vs. model predictions for one or more experiments.

        Parameters
        ----------
        x : array_like
            Parameter vector (e.g. ``result.x``).
        experiments : list[int], optional
            Experiment indices to plot; defaults to all experiments with
            measurements.
        overlay : bool
            If ``False`` (default), one subplot per experiment.
            If ``True``, all on a single axis.
        show_measurements : bool
            Draw measured data points.
        show_predictions : bool
            Draw predicted curves (runs the simulation for each experiment).
        prediction_style : str, optional
            Matplotlib line style for predictions (default: ``"-"``).
        measurement_style : str, optional
            Matplotlib marker style for measurements (default: ``"o"``).
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw into.
        axes : matplotlib.axes.Axes or list, optional
            Existing axes to draw into.
        title : str, optional
            Subplot / figure title.
        xlabel : str
            Time-axis label.
        ylabel : str
            Output-axis label.
        grid : bool
            Draw grid lines.
        legend : bool
            Show legend.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : list[matplotlib.axes.Axes]
        """
        import matplotlib.pyplot as plt  # lazy import

        if not self.experiments:
            raise ValueError("No experiments configured to plot.")

        if experiments is None:
            experiments = [
                i for i, exp in enumerate(self.experiments) if exp.measurements
            ]
        if not experiments:
            raise ValueError("No experiments with measurements to plot.")

        # Build axes grid
        if overlay:
            if axes is None:
                fig = fig or plt.figure(figsize=(8, 5))
                ax = fig.gca()
            else:
                ax = axes
            axes_list = [ax]
        else:
            if axes is None:
                n = len(experiments)
                fig, axes_arr = plt.subplots(
                    n, 1, sharex=True, figsize=(8, max(3, 3 * n))
                )
                axes_list = [axes_arr] if n == 1 else list(axes_arr)
            else:
                axes_list = list(axes)

        meas_style = measurement_style or "o"
        pred_style = prediction_style or "-"

        # Apply parameters once; run each experiment once and read all outputs
        self.apply(x)

        for row, exp_idx in enumerate(experiments):
            exp = self.experiments[exp_idx]
            ax = axes_list[0] if overlay else axes_list[row]

            if show_measurements and exp.measurements:
                for j, meas in enumerate(exp.measurements):
                    name = getattr(meas, "name", None) or f"meas{j}"
                    label = (
                        f"{name} (exp{exp_idx})" if overlay else name
                    )
                    ax.plot(
                        meas.time, meas.data, meas_style,
                        ms=5, alpha=0.6, label=label,
                    )

            if show_predictions and exp.outputs:
                try:
                    exp.runner.run()
                    n_out = len(exp.outputs)
                    for j, out_sig in enumerate(exp.outputs):
                        resolved = self._resolve_output(exp, out_sig)
                        t_pred, y_pred = resolved.read()
                        if overlay:
                            label = f"fit{j} (exp{exp_idx})" if n_out > 1 else f"fit (exp{exp_idx})"
                        else:
                            label = f"fit{j}" if n_out > 1 else "fit"
                        ax.plot(t_pred, y_pred, pred_style, lw=2, label=label)
                except Exception as e:
                    ax.text(
                        0.01, 0.99,
                        f"Prediction failed for exp{exp_idx}:\n"
                        f"{type(e).__name__}: {e}",
                        transform=ax.transAxes,
                        va="top", ha="left", fontsize=9,
                    )

            if not overlay:
                ax.set_title(
                    f"Experiment {exp_idx}" if title is None else title
                )
            ax.set_ylabel(ylabel)
            if grid:
                ax.grid(True, alpha=0.3)
            if legend:
                ax.legend()

        axes_list[-1].set_xlabel(xlabel)
        if overlay:
            axes_list[0].set_title(title or "Fit vs. measurements")

        return fig, axes_list


# HELPER FUNCTIONS ======================================================================

def block_param_to_var(
    block,
    param_name: str,
    value: float | None = None,
    bounds: tuple[float, float] = (-np.inf, np.inf),
    param_id: str | None = None,
    transform: Callable[[float], float] | None = None,
) -> Parameter:
    """Create a block-bound :class:`Parameter` for estimation.

    Parameters
    ----------
    block : pathsim.blocks.Block
        Target block.
    param_name : str
        Attribute name on the block.
    value : float, optional
        Initial optimizer-space value; defaults to the current attribute value.
    bounds : tuple[float, float]
        Lower / upper bounds.
    param_id : str, optional
        Identifier prefix for the parameter name.
    transform : callable, optional
        Optimizer-to-model transform.

    Returns
    -------
    Parameter
    """
    if not hasattr(block, param_name):
        raise AttributeError(f"Block '{block}' has no attribute '{param_name}'")

    name = (
        f"{param_id}.{param_name}"
        if param_id is not None
        else f"{type(block).__name__}.{param_name}"
    )
    if value is None:
        value = getattr(block, param_name)

    return BlockParameter(
        block=block,
        attribute=param_name,
        name=name,
        value=value,
        bounds=bounds,
        transform=transform,
    )


def free_param_to_var(
    param_name: str,
    value: float | None = None,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> Parameter:
    """Create a free (non-block) :class:`Parameter` for estimation.

    Parameters
    ----------
    param_name : str
        Parameter name.
    value : float
        Initial optimizer-space value (required).
    bounds : tuple[float, float]
        Lower / upper bounds.

    Returns
    -------
    Parameter
    """
    if value is None:
        raise ValueError("Initial value must be provided for free parameters.")
    return FreeParameter(name=param_name, value=value, bounds=bounds)
