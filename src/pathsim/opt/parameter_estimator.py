"""
#########################################################################################
##
##                 PARAMETER ESTIMATION TOOLKIT (PathSim Extension Layer)
##                           (parameter_estimator.py)
##
##                              Kevin McBride 2026
##
#########################################################################################

OVERVIEW
--------
This module provides a lightweight parameter-estimation layer for PathSim.

It is intended for iterative optimization loops where a PathSim `Simulation`
is evaluated repeatedly with different parameter values.

FEATURES
--------
- Parameter declaration with optional bounds and transforms
- Measurement handling via time-aligned `TimeSeriesData`
- Model output extraction from PathSim `Scope` via `ScopeSignal`
- Stateful simulation reset + post-reset hooks via `SimRunner`
- SciPy-based fitting (`least_squares` and `minimize`)
- Currently supports multiple experiments with local and global parameters

NOTES
-----
- PathSim simulations are stateful; objective evaluation must reset and rerun.
- Outputs must be read from a `Scope` (or compatible `.read()` provider).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import copy

import numpy as np
import scipy.optimize as sci_opt

from ..blocks._block import Block
from .timeseries_data import TimeSeriesData

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
    "block_param_to_var",
    "free_param_to_var",
]

# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER DECLARATION
# ═════════════════════════════════════════════════════════════════════════════

class Parameter:
    
    """Unified estimation parameter.

    A parameter can either be:
    - a free scalar used in user-defined Python code (e.g., ODE closures)
    - a block-bound parameter mapped to a PathSim block attribute

    Optional transforms allow fitting in an unconstrained space while applying a
    physically valid value to the model (e.g., `np.exp` to enforce positivity).

    Parameters
    ----------
    name : str
        Parameter identifier.
    value : float
        Initial value in optimizer space.
    bounds : tuple[float, float]
        Lower/upper bounds in optimizer space.
    transform : callable, optional
        Value transform applied when the parameter is read/applied.
    block : object, optional
        Target block/object for block-bound parameters.
    attribute : str, optional
        Target attribute path (supports dotted access).

    Notes
    -----
    For block-bound parameters, calling `set(...)` applies the transformed value
    to the target object immediately.
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
        self.bounds = bounds
        self.transform = transform
        
        # Block parameter attributes
        self.block = block
        self.attribute = attribute
        
        # Validate configuration
        if block is not None and attribute is None:
            raise ValueError("attribute must be provided when block is specified")
        
        # Determine type
        self._is_block_param = block is not None
        
        # Initialize - apply to block if needed
        self.set(value)
    
    
    @property
    def is_block_parameter(self) -> bool:
        """Check if this is a block parameter."""
        return self._is_block_param
    
    
    @property
    def is_free_parameter(self) -> bool:
        """Check if this is a free parameter."""
        return not self._is_block_param
    
    
    @property
    def value(self) -> float:
        """Current parameter value (in optimizer space)."""
        return self._value
    
    
    @value.setter
    def value(self, new_value: float) -> None:
        self.set(new_value)
    
    
    def __call__(self) -> float:
        """Return the parameter value in model space (after optional transform)."""
        return self.transform(self._value) if self.transform is not None else self._value
    
    
    def set(self, value: float) -> None:
        """Set optimizer-space value and push to target (if block-bound)."""
        self._value = float(value)
        transformed = self()
        
        if self._is_block_param:
            # Apply to block attribute
            obj = self.block
            attrs = self.attribute.split('.')
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], transformed)
    
    
    def __repr__(self) -> str:
        if self._is_block_param:
            return (f"Parameter(name={self.name!r}, value={self._value}, "
                    f"block={type(self.block).__name__}, "
                    f"attribute={self.attribute!r}, bounds={self.bounds})")
        else:
            return (f"Parameter(name={self.name!r}, value={self._value}, "
                    f"bounds={self.bounds})")


#TODO: remove these in future version?
def BlockParameter(block, attribute, name=None, **kwargs):
    """Factory for block-bound Parameters.

    Parameters
    ----------
    block : object
        Target block/object.
    attribute : str
        Attribute path (e.g. "value" or "config.gain").

    Returns
    -------
    Parameter
        A block-bound parameter.
    """
    if name is None:
        name = f'{type(block).__name__}.{attribute}'
    return Parameter(name=name, block=block, attribute=attribute, **kwargs)


def FreeParameter(name, **kwargs):
    """Factory for free (non-block) Parameters."""
    return Parameter(name=name, **kwargs)



class SharedBlockParameter(Parameter):

    """A block-bound parameter applied to the same attribute on multiple targets.

    This is intended for multi-experiment fitting where each experiment uses a
    deep-copied Simulation, but the fitted parameter should be shared globally.

    Notes
    -----
    - `targets` is a list of block objects (one per experiment) that all receive
      the same transformed value.
    - `block` is left as the first target to preserve existing repr/debug.
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
        self.targets = targets
        super().__init__(
            name=name,
            value=value,
            bounds=bounds,
            transform=transform,
            block=targets[0],
            attribute=attribute,
        )

    def set(self, value: float) -> None:
        self._value = float(value)
        transformed = self()

        for tgt in self.targets:
            obj = tgt
            attrs = self.attribute.split('.')
            for attr in attrs[:-1]:
                obj = getattr(obj, attr)
            setattr(obj, attrs[-1], transformed)


@dataclass
class ScopeSignal:
    
    """Scope output selection.

    Reads a single port from an object that provides `read() -> (t, y)`.

    Parameters
    ----------
    scope : object
        Scope-like provider with `.read()`.
    port : int
        Port index to extract from multi-port scopes.

    Notes
    -----
    For multi-experiment fitting with deep-copied simulations, storing a direct
    `scope` reference can accidentally bind to experiment 0's scope. To support
    resolving the matching scope inside each experiment, a ScopeSignal can also
    carry (block_type, occurrence) identifiers.
    """

    scope: object | None = None
    port: int = 0

    # Optional selector to resolve the correct scope instance per experiment
    block_type: str | None = None
    occurrence: int | None = None


    def _extract(self, t, y):
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 1:
            # Single signal already
            y_port = y_arr
        elif y_arr.ndim == 2:
            # Support (n_ports, n_samples) and (n_samples, n_ports)
            if y_arr.shape[0] == t_arr.size and y_arr.shape[1] != t_arr.size:
                y_arr = y_arr.T
            y_port = y_arr[self.port, :]
        else:
            raise ValueError("ScopeSignal supports 1D or 2D scope data only")

        return t_arr, np.asarray(y_port, dtype=float).reshape(-1)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Read `(t, y)` from the scope and extract the selected port."""
        if self.scope is None:
            raise ValueError("ScopeSignal.scope is None; it must be resolved before reading")
        t, y = self.scope.read()
        return self._extract(t, y)


@dataclass
class SimRunner:
    
    """Simulation runner adapter.

    Provides a callable/reset+run interface suitable for optimizers.

    Parameters
    ----------
    sim : object
        PathSim simulation instance.
    output : ScopeSignal
        Output extractor (typically used outside `run()`).
    duration : float
        Simulation duration per evaluation.
    post_reset_hooks : list[callable], optional
        Hooks executed immediately after `sim.reset(...)`.
    pre_run : callable, optional
        Hook executed before `sim.run(...)`.
    adaptive : bool
        Enable adaptive stepping (if supported by PathSim simulation).
    reset_time : float
        Reset time passed into `sim.reset(time=...)`.
    suppress_reset_log : bool
        Temporarily disable sim logging during reset/run for cleaner optimization output.
    """

    sim: object
    output: ScopeSignal
    duration: float
    post_reset_hooks: list[Callable[[], None]] | None = None
    pre_run: Callable[[], None] | None = None
    adaptive: bool = False
    reset_time: float = 0.0
    suppress_reset_log: bool = True


    def after_reset(self, fn: Callable[[], None]) -> "SimRunner":
        """Register a hook executed after every `sim.reset(...)`."""
        if self.post_reset_hooks is None:
            self.post_reset_hooks = []
        self.post_reset_hooks.append(fn)
        return self


    def before_run(self, fn: Callable[[], None]) -> "SimRunner":
        """Set a hook executed immediately before `sim.run(...)`."""
        self.pre_run = fn
        return self


    def __call__(self, _x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the simulation and return the configured output."""
        self.run()
        return self.output.read()


    def run(self) -> None:
        """Reset and run the simulation (no output read)."""
        old_log = None
        if self.suppress_reset_log and hasattr(self.sim, 'log'):
            old_log = self.sim.log
            self.sim.log = False

        try:        
            self.sim.reset(time=self.reset_time)

            # Run after_reset hooks
            if self.post_reset_hooks is not None:
                for hook in self.post_reset_hooks:
                    hook()

            if self.pre_run is not None:
                self.pre_run()

            self.sim.run(duration=self.duration, reset=False, adaptive=self.adaptive)
        finally:
            # Restore original logging state
            if old_log is not None:
                self.sim.log = old_log


@dataclass
class Experiment:

    """A single experiment (one sim/run) with one or more datasets.

    Each experiment is evaluated by running its runner once per objective call,
    then comparing one or more measurements to one or more mapped scope signals.

    Parameters
    ----------
    runner : SimRunner
        Runner responsible for resetting and running the simulation.
    duration : float, optional
        Optional explicit duration override for this experiment.
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


@dataclass
class EstimatorResult:
    
    """Optimization result container."""
    x: np.ndarray
    cost: float
    nfev: int
    success: bool
    message: str


class ParameterEstimator:
    
    """Parameter estimation driver.

    Parameters
    ----------
    parameters : list[Parameter], optional
        Parameters to estimate.
    simulator : object
        PathSim simulation or a runner providing `run()` / callable interface.
    measurement : TimeSeriesData | list[TimeSeriesData], optional
        Measurement(s) to fit.
    outputs : ScopeSignal | list[ScopeSignal], optional
        Output mapping(s) corresponding to measurements.
    duration : float, optional
        Override simulation duration; otherwise derived from measurements.
    adaptive : bool
        Enable adaptive stepping (if supported).
    pre_run : callable, optional
        Hook executed before each sim run.
    sigma : float | np.ndarray | list, optional
        Measurement noise scaling for residual normalization.
    """

    def __init__(
        self,
        parameters: list[Parameter] | None = None,
        simulator=None,
        measurement=None,
        outputs=None,
        duration: float | None = None,
        adaptive: bool = False,
        pre_run: Callable | None = None,
        sigma: float | np.ndarray | None = None,
    ):
        if parameters is None:
            parameters = []

        # Global/shared parameters (applied to all experiments)
        self.global_parameters: list[Parameter] = []
        # Local parameters per experiment (applied only to that experiment)
        self.local_parameters: list[list[Parameter]] = []

        # Backwards compatibility: treat provided parameters list as global
        self.global_parameters.extend(parameters)

        # Prefer experiments as the primary internal structure (supports multi-run fitting)
        self.experiments: list[Experiment] = []

        # Keep a reference simulator for cloning experiments (deepcopy)
        self._base_simulator = simulator if simulator is not None and hasattr(simulator, 'run') else None

        # Remember defaults for auto-created experiments
        self._default_runner_kwargs = dict(
            adaptive=adaptive,
            pre_run=pre_run,
        )

        # Backwards compatible: create experiment 0 from simulator if provided
        if simulator is not None:
            self.add_experiment(
                simulator,
                duration=duration,
                adaptive=adaptive,
                pre_run=pre_run,
            )

        # Backwards compatible: accept measurement/outputs/sigma in __init__ and attach to experiment 0
        if measurement is None:
            measurement = []

        if isinstance(measurement, TimeSeriesData):
            measurement_list = [measurement]
            outputs_list = [outputs] if outputs is not None else [None]
        else:
            measurement_list = list(measurement)
            outputs_list = list(outputs) if outputs is not None else [None] * len(measurement_list)

        if sigma is None:
            sigma_list: list[float | None] = [None] * len(measurement_list)
        elif isinstance(sigma, (list, np.ndarray)):
            sigma_list = [float(s) if s is not None else None for s in sigma]
        else:
            sigma_list = [float(sigma)] * len(measurement_list)

        if measurement_list:
            if not self.experiments:
                raise ValueError("measurement provided but no simulator/runner configured")

            for ts, out, sig in zip(measurement_list, outputs_list, sigma_list):
                if out is None:
                    raise ValueError(
                        "outputs must be provided when passing measurement(s) into ParameterEstimator.__init__"
                    )

                if isinstance(out, ScopeSignal):
                    scope = out.scope
                    port = int(out.port)
                else:
                    scope, port = self._scope_port_from_signal(out)

                self.add_timeseries(ts, scope=scope, port=port, sigma=sig, experiment=0)

        # Legacy aliases for downstream compatibility
        self.runners = [exp.runner for exp in self.experiments] if self.experiments else []

        # Derive durations after initial dataset registration
        self.duration = 0.0
        self._update_duration_from_measurements()


    @property
    def parameters(self) -> list[Parameter]:
        """Flattened parameter list in optimizer order: globals then locals by experiment."""
        params: list[Parameter] = []
        params.extend(self.global_parameters)
        for exp_params in self.local_parameters:
            params.extend(exp_params)
        return params


    def _rebuild_local_parameter_container(self) -> None:
        """Ensure local parameter list has one entry per experiment."""
        while len(self.local_parameters) < len(self.experiments):
            self.local_parameters.append([])


    def _update_duration_from_measurements(self) -> None:
        """Update experiment runner durations from their measurements.

        Explicit durations passed to :meth:`add_experiment` are preserved;
        measurement-derived durations are used only as a floor.
        """
        for exp in self.experiments:
            if not exp.measurements:
                continue

            derived = float(max(meas.time.max() for meas in exp.measurements))

            if exp.duration is not None:
                dur = max(float(exp.duration), derived)
            else:
                dur = derived

            if hasattr(exp.runner, "duration"):
                exp.runner.duration = dur

        all_meas = [m for exp in self.experiments for m in (exp.measurements or [])]
        self.duration = float(max((m.time.max() for m in all_meas), default=0.0))


    def add_experiment(
        self,
        simulator,
        *,
        duration: float | None = None,
        adaptive: bool = False,
        pre_run: Callable[[], None] | None = None,
        reset_time: float = 0.0,
        suppress_reset_log: bool = True,
        copy_sim: bool = False,
    ) -> int:
        """Register a new experiment and return its index.

        Parameters
        ----------
        simulator : object
            PathSim simulation (has .run/.reset) or runner-like object.
        copy_sim : bool
            If True and `simulator` is a PathSim Simulation, deep-copies it so each
            experiment has independent state.
        """
        if hasattr(simulator, 'run'):
            sim_obj = copy.deepcopy(simulator) if copy_sim else simulator
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

        self.experiments.append(Experiment(runner=runner, duration=float(duration) if duration is not None else None))
        self.runners = [exp.runner for exp in self.experiments]

        # keep locals aligned
        self._rebuild_local_parameter_container()

        self._update_duration_from_measurements()
        return len(self.experiments) - 1


    def _ensure_experiment(self, idx: int) -> None:
        """Ensure that experiment indices up to `idx` exist.

        If a base simulator was provided at construction time, missing experiments
        are deep-copied from it.
        """
        if idx < 0:
            raise IndexError("experiment index must be >= 0")

        while len(self.experiments) <= idx:
            if self._base_simulator is None:
                raise IndexError(
                    f"experiment index {idx} out of range and no base simulator is available for cloning. "
                    "Call add_experiment(...) explicitly."
                )
            self.add_experiment(
                self._base_simulator,
                copy_sim=True,
                **self._default_runner_kwargs,
            )

        self._rebuild_local_parameter_container()


    @staticmethod
    def _block_occurrence(sim, block_obj: object) -> tuple[str, int]:
        """Return (type_name, occurrence_index) for a block object within sim.blocks."""
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
        """Find the Nth block of a given class name in sim.blocks."""
        occ = 0
        for b in sim.blocks:
            if type(b).__name__ == type_name:
                if occ == occurrence:
                    return b
                occ += 1
        raise ValueError(f"No block '{type_name}' occurrence {occurrence} found in simulation")


    def _resolve_output(self, exp: Experiment, sig: ScopeSignal) -> ScopeSignal:
        """Resolve a ScopeSignal to the matching scope object inside an experiment sim."""
        if sig.scope is not None:
            return sig

        sim = getattr(exp.runner, 'sim', None)
        if sim is None:
            raise ValueError("Cannot resolve ScopeSignal without a SimRunner-backed experiment")

        if sig.block_type is None or sig.occurrence is None:
            raise ValueError("Unresolvable ScopeSignal (missing selector and scope reference)")

        scope_obj = self._find_block_by_occurrence(sim, sig.block_type, sig.occurrence)
        return ScopeSignal(scope=scope_obj, port=int(sig.port), block_type=sig.block_type, occurrence=sig.occurrence)


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
        """Add a measurement and map it to a scope port for a given experiment.

        Parameters
        ----------
        ts : TimeSeriesData
            Measurement dataset.
        scope : object, optional
            Scope block with ``.read()`` method.
        port : int, optional
            Port index on the scope.
        signal : object, optional
            PortReference (e.g. ``scope[0]``). Pass either ``signal=`` or
            ``scope=``/``port=``, not both.
        sigma : float, optional
            Measurement noise scaling for residual normalization.
        experiment : int
            Experiment index to attach this dataset to.
        """
        if not isinstance(ts, TimeSeriesData):
            raise TypeError(f"add_timeseries expects TimeSeriesData, got {type(ts).__name__}")

        self._ensure_experiment(experiment)

        if signal is not None:
            if scope is not None or port is not None:
                raise ValueError("Pass either `signal=` OR (`scope=`, `port=`), not both.")
            scope, port = self._scope_port_from_signal(signal)

        if scope is None or port is None:
            raise ValueError("You must pass either scope=..., port=... or signal=scope[0].")

        # For multi-experiment fitting, store a (type, occurrence) selector so the
        # correct deep-copied scope can be resolved per experiment at run time.
        block_type = None
        occurrence = None
        sim0 = getattr(self.experiments[0].runner, 'sim', None) if self.experiments else None
        if sim0 is not None:
            try:
                block_type, occurrence = self._block_occurrence(sim0, scope)
            except Exception:
                block_type, occurrence = None, None

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

        self._update_duration_from_measurements()
        return self


    def add_block_parameter(
        self,
        block,
        param_name,
        value=None,
        bounds=(-np.inf, np.inf),
        param_id=None,
        transform: Callable[[float], float] | None = None,
    ) -> "ParameterEstimator":
        """Add a block-bound parameter as a global parameter.

        For multi-experiment shared parameters, use :meth:`add_global_block_parameter`.

        Parameters
        ----------
        block : object
            Target block instance.
        param_name : str
            Attribute name on the block.
        value : float, optional
            Initial optimizer-space value. Defaults to current attribute value.
        bounds : tuple of float
            ``(lower, upper)`` bounds in optimizer space.
        param_id : str, optional
            Human-readable prefix for the parameter name.
        transform : callable, optional
            Mapping from optimizer space to model space (e.g. ``np.exp``).
        """
        param = block_param_to_var(
            block,
            param_name,
            value=value,
            bounds=bounds,
            param_id=param_id,
            transform=transform,
        )
        self.global_parameters.append(param)
        return self


    def add_shared_block_parameter(
        self,
        block_name: str,
        param_name: str,
        *,
        value: float | None = None,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        param_id: str | None = None,
        transform: Callable[[float], float] | None = None,
    ) -> "ParameterEstimator":
        """Add a single global parameter applied to the same block attribute in every experiment.

        Parameters
        ----------
        block_name : str
            Name of the block class to match (e.g. "Constant", "Integrator").
            The first instance of that class found in each experiment sim is used.
        param_name : str
            Attribute name on the block.

        Notes
        -----
        This is a convenience helper for deep-copied sims where the same parameter
        should be shared across all experiments.
        """
        if not self.experiments:
            raise ValueError("No experiments configured. Pass simulator=... or call add_experiment().")

        targets: list[Any] = []
        for exp in self.experiments:
            sim = getattr(exp.runner, 'sim', None)
            if sim is None:
                raise ValueError("Shared block parameters require SimRunner experiments.")

            match = None
            for b in sim.blocks:
                if type(b).__name__ == block_name:
                    match = b
                    break
            if match is None:
                raise ValueError(f"Experiment sim has no block of type '{block_name}'.")
            if not hasattr(match, param_name):
                raise AttributeError(f"Block '{block_name}' has no attribute '{param_name}'")
            targets.append(match)

        pname = f"{param_id}.{param_name}" if param_id is not None else f"{block_name}.{param_name}"
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
        return self


    def add_parameters(self, params) -> "ParameterEstimator":
        """Backwards compatible: adds parameters as global."""
        self.global_parameters.extend(params)
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
        """Alias for shared/global block parameter across all experiments."""
        return self.add_shared_block_parameter(
            block_name,
            param_name,
            value=value,
            bounds=bounds,
            param_id=param_id,
            transform=transform,
        )


    def add_local_block_parameter(
        self,
        experiment: int,
        block_name: str,
        param_name: str,
        *,
        value: float | None = None,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        param_id: str | None = None,
    ) -> "ParameterEstimator":
        """Add an experiment-local block parameter (distinct value per experiment)."""
        self._ensure_experiment(experiment)

        exp = self.experiments[experiment]
        sim = getattr(exp.runner, 'sim', None)
        if sim is None:
            raise ValueError("Local block parameters require SimRunner experiments.")

        match = None
        for b in sim.blocks:
            if type(b).__name__ == block_name:
                match = b
                break
        if match is None:
            raise ValueError(f"Experiment {experiment} sim has no block of type '{block_name}'.")
        if not hasattr(match, param_name):
            raise AttributeError(f"Block '{block_name}' has no attribute '{param_name}'")

        if param_id is None:
            pname = f"exp{experiment}.{block_name}.{param_name}"
        else:
            pname = f"{param_id}.exp{experiment}.{param_name}"

        if value is None:
            value = float(getattr(match, param_name))

        self.local_parameters[experiment].append(
            BlockParameter(
                block=match,
                attribute=param_name,
                name=pname,
                value=float(value),
                bounds=bounds,
            )
        )
        return self


    @staticmethod
    def _scope_port_from_signal(signal: object) -> tuple[object, int]:
        """Extract `(scope, port)` mapping.

        Supports:
        - explicit `(scope, port)` tuples
        - PathSim PortReference-like objects with `.block` and `.ports`
        """
        if isinstance(signal, (tuple, list)) and len(signal) == 2:
            return signal[0], int(signal[1])

        blk = getattr(signal, "block", None)
        ports = getattr(signal, "ports", None)
        if blk is not None and ports is not None:
            if len(ports) != 1:
                raise ValueError(f"Expected PortReference with a single port, got {len(ports)}.")
            return blk, int(ports[0])

        raise TypeError(
            f"Unsupported signal type {type(signal).__name__}. "
            "Use signal=scope[0] (PortReference) or signal=(scope, port)."
        )


    def simulate(self, x: np.ndarray, output_idx: int = 0, *, experiment: int = 0):
        """Simulate a selected experiment and return one of its mapped outputs.

        Backwards compatible with legacy signature: simulate(x, output_idx=0)
        """
        if not self.experiments:
            raise ValueError("No experiments configured.")
        if experiment < 0 or experiment >= len(self.experiments):
            raise IndexError(f"experiment index {experiment} out of range (0..{len(self.experiments)-1})")

        self._update_duration_from_measurements()
        self.apply(x)

        exp = self.experiments[experiment]
        if hasattr(exp.runner, "run") and callable(getattr(exp.runner, "run")):
            exp.runner.run()

            if output_idx < 0 or output_idx >= len(exp.outputs):
                raise IndexError(f"output_idx {output_idx} out of range for experiment {experiment}")

            out = self._resolve_output(exp, exp.outputs[output_idx])
            return out.read()

        exp.runner.reset(time=0.0)
        return exp.runner(None)


    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute stacked residual vector across all experiments and datasets."""
        if not self.experiments:
            raise ValueError("No experiments configured.")

        self._update_duration_from_measurements()
        self.apply(x)

        all_residuals: list[np.ndarray] = []

        for exp_idx, exp in enumerate(self.experiments):
            if not exp.measurements:
                continue

            # Run sim once per experiment
            if hasattr(exp.runner, "run") and callable(getattr(exp.runner, "run")):
                exp.runner.run()
            else:
                # Callable runner path: only supports a single dataset for that experiment.
                if len(exp.measurements) != 1:
                    raise ValueError(
                        f"Experiment {exp_idx} uses a callable runner; only 1 dataset is supported for that experiment."
                    )

                t_sim, y_sim = exp.runner(None)
                t_sim = np.asarray(t_sim, dtype=float).reshape(-1)
                y_sim = np.asarray(y_sim, dtype=float).reshape(-1)

                meas = exp.measurements[0]
                y_pred = np.interp(meas.time, t_sim, y_sim)
                sigma_i = exp.sigma[0] if exp.sigma[0] is not None else 1.0
                all_residuals.append((y_pred - meas.data) / sigma_i)
                continue

            # Evaluate all datasets mapped to this experiment
            if len(exp.outputs) != len(exp.measurements):
                raise ValueError(
                    f"Experiment {exp_idx} has {len(exp.measurements)} measurement(s) but {len(exp.outputs)} output mapping(s)."
                )

            for i, meas in enumerate(exp.measurements):
                out = self._resolve_output(exp, exp.outputs[i])

                t_out, y_out = out.read()
                t_out = np.asarray(t_out, dtype=float).reshape(-1)
                y_out = np.asarray(y_out, dtype=float).reshape(-1)

                y_pred = np.interp(meas.time, t_out, y_out)
                sigma_i = exp.sigma[i] if exp.sigma[i] is not None else 1.0
                all_residuals.append((y_pred - meas.data) / sigma_i)

        if not all_residuals:
            return np.array([], dtype=float)

        return np.concatenate(all_residuals)


    def display(self) -> None:
        """Print a summary of all parameters and their current values."""
        print("=" * 60)
        print("Parameter Estimation Results")
        print("=" * 60)

        if self.global_parameters:
            print("\nGlobal parameters:")
            print("-" * 40)
            for p in self.global_parameters:
                transformed = p()
                if p.transform is not None:
                    print(f"  {p.name:30s}  x={p.value:.6g}  ->  {transformed:.6g}")
                else:
                    print(f"  {p.name:30s}  = {transformed:.6g}")

        for i, exp_params in enumerate(self.local_parameters):
            if exp_params:
                print(f"\nLocal parameters (experiment {i}):")
                print("-" * 40)
                for p in exp_params:
                    transformed = p()
                    if p.transform is not None:
                        print(f"  {p.name:30s}  x={p.value:.6g}  ->  {transformed:.6g}")
                    else:
                        print(f"  {p.name:30s}  = {transformed:.6g}")

        print("=" * 60)


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
        """Fit parameters using SciPy optimizers.

        Parameters
        ----------
        x0 : sequence of float, optional
            Initial optimizer-space parameter vector.
        bounds : (lower, upper), optional
            Bounds in optimizer space.
        loss : str
            Loss for `scipy.optimize.least_squares`.
        f_scale : float
            Loss scale for robust losses.
        max_nfev : int
            Max function evaluations / iterations (depending on solver).
        verbose : int
            Verbosity level.
        method : str
            Solver selection: "least_squares" or a `scipy.optimize.minimize` method.
        constraints : list of dict, optional
            Constraint definitions for `scipy.optimize.minimize`.

        Notes
        -----
        - General constraints are only supported via `minimize` methods such as
          'SLSQP', 'trust-constr', or 'COBYLA'.
        """
        # Auto-extract x0 and bounds from Parameters
        if x0 is None:
            x0 = [p.value for p in self.parameters]
        x0_arr = np.asarray(x0, dtype=float)
        
        if bounds is None:
            lower = np.array([p.bounds[0] for p in self.parameters], dtype=float)
            upper = np.array([p.bounds[1] for p in self.parameters], dtype=float)
            bounds = (lower, upper)
        
        bounds_list = list(zip(bounds[0], bounds[1]))
        
        
        # Objective function (scalar cost from residuals)
        def objective(x):
            r = self.residuals(x)
            return 0.5 * np.sum(r**2)
        
        
        def _callback(xk):
            if verbose > 0:
                print(f"iter x={xk}, obj={objective(xk)}")
        
        
        # Choose solver
        if method == "least_squares":
            if constraints is not None:
                raise ValueError(
                    "least_squares does not support general constraints. "
                    "Use method='SLSQP' or 'trust-constr' instead."
                )
            
            res = sci_opt.least_squares(
                self.residuals,
                x0=x0_arr,
                bounds=bounds,
                loss=loss,
                f_scale=float(f_scale),
                max_nfev=int(max_nfev),
                verbose=int(verbose),
            )
            
            # self.apply(res.x)
            
            return EstimatorResult(
                x=res.x,
                cost=float(res.cost),
                nfev=int(res.nfev),
                success=bool(res.success),
                message=str(res.message),
            )
        
        elif method in ["SLSQP", "trust-constr", "COBYLA"]:
            # Methods that support constraints
            res = sci_opt.minimize(
                objective,
                x0=x0_arr,
                bounds=bounds_list,
                method=method,
                constraints=constraints,  # ← Pass constraints here
                callback=_callback,
                options={'maxiter': max_nfev, 'disp': verbose > 0}
            )
           
            # self.apply(res.x)
            # Compute final cost
            final_residuals = self.residuals(res.x)
            cost = 0.5 * np.sum(final_residuals**2)
            
            return EstimatorResult(
                x=res.x,
                cost=float(cost),
                nfev=int(res.nfev) if hasattr(res, 'nfev') else max_nfev,
                success=bool(res.success),
                message=str(res.message),
            )
        
        else:
            # Other methods (L-BFGS-B, differential_evolution, etc.)
            if constraints is not None:
                raise ValueError(
                    f"Method '{method}' does not support general constraints. "
                    "Use 'SLSQP', 'trust-constr', or 'COBYLA'."
                )
            
            res = sci_opt.minimize(
                objective,
                x0=x0_arr,
                bounds=bounds_list,
                method=method,
                callback=_callback,
                options={'maxiter': max_nfev, 'disp': True}
            )
            
            # self.apply(res.x)
            
            final_residuals = self.residuals(res.x)
            cost = 0.5 * np.sum(final_residuals**2)
            
            return EstimatorResult(
                x=res.x,
                cost=float(cost),
                nfev=int(res.nfev) if hasattr(res, 'nfev') else max_nfev,
                success=bool(res.success),
                message=str(res.message),
            )
    
    def apply(self, x: np.ndarray) -> None:
        """Apply optimizer parameter vector to the model(s).

        Optimizer vector order:
        - global parameters (shared across experiments)
        - local parameters for exp0
        - local parameters for exp1
        - ...
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

        # Update source-only blocks (no inputs) after parameter changes for all experiments.
        for exp in self.experiments:
            sim = getattr(exp.runner, 'sim', None)
            if sim is None:
                continue
            for block in sim.blocks:
                if hasattr(block, 'num_inputs') and block.num_inputs == 0:
                    block.update(0.0)


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
        """Plot measurements vs model prediction(s) for one or more experiments.

        Parameters
        ----------
        x:
            Parameter vector (e.g. fit.x).
        experiments:
            Experiment indices to plot. Defaults to all experiments that have datasets.
        overlay:
            If False (default), create one subplot per experiment.
            If True, plot all experiments on the same axis.
        show_measurements / show_predictions:
            Toggle drawing measured points and predicted curves.
        prediction_style / measurement_style:
            Optional Matplotlib style strings, e.g. '-' or 'o'.
        fig / axes:
            Pass an existing figure/axes to draw into.
        """
        # Lazy import so core estimator doesn't require matplotlib at import-time
        import matplotlib.pyplot as plt

        if not hasattr(self, "experiments") or not self.experiments:
            raise ValueError("No experiments configured to plot.")

        # Default experiments: only those with measurements
        if experiments is None:
            experiments = [i for i, exp in enumerate(self.experiments) if getattr(exp, "measurements", None)]
        if not experiments:
            raise ValueError("No experiments with measurements to plot.")

        # Create axes
        if overlay:
            if axes is None:
                fig = fig if fig is not None else plt.figure(figsize=(8, 5))
                ax = fig.gca()
            else:
                ax = axes
            axes_list = [ax]
        else:
            if axes is None:
                n = len(experiments)
                fig, axes_list = plt.subplots(n, 1, sharex=True, figsize=(8, max(3, 3 * n)))
                if n == 1:
                    axes_list = [axes_list]
            else:
                # allow caller to pass list/array of axes
                axes_list = list(axes)

        # Helper for consistent styling
        meas_style = measurement_style if measurement_style is not None else "o"
        pred_style = prediction_style if prediction_style is not None else "-"

        # Apply parameters once here; simulate() will run each experiment
        # (simulate() should call apply(x) internally too; calling here is harmless but optional)
        # self.apply(x)

        for row, exp_idx in enumerate(experiments):
            exp = self.experiments[exp_idx]
            ax = axes_list[0] if overlay else axes_list[row]

            # Measurements for this experiment
            if show_measurements and getattr(exp, "measurements", None):
                for j, meas in enumerate(exp.measurements):
                    name = getattr(meas, "name", None) or f"meas{j}"
                    ax.plot(
                        meas.time,
                        meas.data,
                        meas_style,
                        ms=5,
                        alpha=0.6,
                        label=f"{name} (exp{exp_idx})" if overlay else f"{name}",
                    )

            # Prediction: for now plot output_idx=0 for this experiment
            # If you want multiple outputs per experiment, extend with output_idx list.
            if show_predictions:
                try:
                    t_pred, y_pred = self.simulate(x, experiment=exp_idx)
                    ax.plot(
                        t_pred,
                        y_pred,
                        pred_style,
                        lw=2,
                        label=f"fit (exp{exp_idx})" if overlay else "fit",
                    )
                except Exception as e:
                    # Don’t hard-crash plotting if one exp fails; let user see others
                    ax.text(
                        0.01,
                        0.99,
                        f"Prediction failed for exp{exp_idx}:\n{type(e).__name__}: {e}",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=9,
                    )

            if not overlay:
                ax.set_title(f"Experiment {exp_idx}" if title is None else title)

            ax.set_ylabel(ylabel)
            if grid:
                ax.grid(True, alpha=0.3)

            if legend:
                ax.legend()

        # Axis labels / title
        axes_list[-1].set_xlabel(xlabel)
        if overlay and title is not None:
            axes_list[0].set_title(title)
        elif overlay and title is None:
            axes_list[0].set_title("Fit vs measurements")

        return fig, axes_list


def free_param_to_var(param_name, value=None, bounds=(-np.inf, np.inf)):
    """Create a free (non-block) parameter for estimation.

    Parameters
    ----------
    param_name : str
        Parameter name.
    value : float
        Initial value (optimizer space).
    bounds : tuple of float
        ``(lower, upper)`` bounds.

    Returns
    -------
    Parameter
        Free parameter instance.
    """
    if value is None:
        raise ValueError("Initial value must be provided for free parameters.")

    return FreeParameter(
        name=param_name,
        value=value,
        bounds=bounds,
    )   


def block_param_to_var(block, param_name, value=None, bounds=(-np.inf, np.inf), param_id=None, transform=None):
    """Create a block-bound parameter for estimation.

    Parameters
    ----------
    block : pathsim.blocks.Block
        Target block.
    param_name : str
        Attribute name on the block.
    value : float, optional
        Initial value; defaults to current block attribute.
    bounds : tuple[float, float]
        Lower/upper bounds.
    param_id : str, optional
        Identifier prefix for the parameter name.
    transform : callable, optional
        Mapping from optimizer space to model space.

    Returns
    -------
    Parameter
        Block-bound parameter instance.
    """

    if not hasattr(block, param_name):
        raise AttributeError(f"Block '{block}' has no attribute '{param_name}'")
    
    if param_id is None:
        name = f"{block.__class__.__name__}.{param_name}"
    else:
        name = f"{param_id}.{param_name}"

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

# NOTE: get_block_dict was removed.
# It relied on the caller's globals() which is unreliable when imported.
# Users should build block lookups explicitly:
#
#   block_dict = {name: obj for name, obj in locals().items()
#                 if isinstance(obj, Block) and obj in sim.blocks}