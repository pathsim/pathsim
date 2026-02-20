########################################################################################
##
##                                  TESTS FOR 
##           'opt/parameter_estimator.py' and 'opt/timeseries_data.py'
##
##                              Kevin McBride 2026
##
########################################################################################

# IMPORTS ==============================================================================


"""
Tests for pathsim.opt.parameter_estimator

Covers:
- Parameter, BlockParameter, FreeParameter, SharedBlockParameter
- ScopeSignal
- SimRunner
- ParameterEstimator (single and multi-experiment fitting)
"""

import numpy as np
import pytest

from pathsim.opt.parameter_estimator import (
    Parameter,
    BlockParameter,
    FreeParameter,
    SharedBlockParameter,
    ScopeSignal,
    SimRunner,
    Experiment,
    EstimatorResult,
    ParameterEstimator,
    block_param_to_var,
    free_param_to_var,
)
from pathsim.opt.timeseries_data import TimeSeriesData


# ═══════════════════════════════════════════════════════════════════════════
# Helpers / Fixtures
# ═══════════════════════════════════════════════════════════════════════════

class _DummyBlock:
    """Minimal block-like object for unit tests."""

    def __init__(self, value=1.0, gain=2.0):
        self.value = value
        self.gain = gain
        self.num_inputs = 0

    def update(self, t):
        pass


class _NestedBlock:
    """Block with dotted attribute path."""

    class Config:
        def __init__(self):
            self.gain = 5.0

    def __init__(self):
        self.config = self.Config()
        self.num_inputs = 0

    def update(self, t):
        pass


class _DummyScope:
    """Minimal scope-like object that stores and returns (t, y)."""

    def __init__(self, t=None, y=None):
        self._t = t if t is not None else np.array([0.0, 1.0])
        self._y = y if y is not None else np.array([0.0, 1.0])

    def read(self):
        return self._t, self._y

    def __getitem__(self, port):
        """Simulate scope[port] -> PortReference."""
        return _PortRef(self, port)


class _PortRef:
    """Minimal PortReference-like object."""

    def __init__(self, block, port):
        self.block = block
        self.ports = [port]


class _DummySim:
    """Minimal Simulation-like object for testing SimRunner / ParameterEstimator."""

    def __init__(self, blocks=None):
        self.blocks = blocks or []
        self.log = False
        self._ran = False
        self._reset_count = 0

    def reset(self, time=0.0):
        self._reset_count += 1

    def run(self, duration=1.0, reset=True, adaptive=False):
        self._ran = True
        # Populate any _DummyScope blocks with some data
        for b in self.blocks:
            if isinstance(b, _DummyScope):
                t = np.linspace(0, duration, 50)
                # Find all _DummyBlock blocks and sum their values as "gain"
                gain = sum(
                    getattr(blk, "value", 0)
                    for blk in self.blocks
                    if isinstance(blk, _DummyBlock)
                )
                if gain == 0:
                    gain = 1.0
                b._t = t
                b._y = gain * t


# ═══════════════════════════════════════════════════════════════════════════
# Parameter tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParameter:

    def test_free_parameter_init(self):
        p = Parameter(name="alpha", value=2.0, bounds=(0, 10))
        assert p.name == "alpha"
        assert p.value == 2.0
        assert p() == 2.0
        assert p.bounds == (0, 10)
        assert p.is_free_parameter
        assert not p.is_block_parameter

    def test_free_parameter_with_transform(self):
        p = Parameter(name="log_k", value=1.0, transform=np.exp)
        assert p.value == 1.0
        assert p() == pytest.approx(np.e)

    def test_block_parameter_applies_to_block(self):
        blk = _DummyBlock(value=10.0)
        p = Parameter(name="blk.value", value=3.0, block=blk, attribute="value")
        assert p.is_block_parameter
        assert blk.value == 3.0  # set() was called in __init__

    def test_block_parameter_with_transform(self):
        blk = _DummyBlock(value=0.0)
        p = Parameter(name="blk.value", value=2.0, block=blk, attribute="value", transform=lambda x: x ** 2)
        assert blk.value == 4.0
        assert p() == 4.0
        assert p.value == 2.0  # optimizer space

    def test_block_parameter_dotted_attribute(self):
        blk = _NestedBlock()
        p = Parameter(name="nested", value=9.0, block=blk, attribute="config.gain")
        assert blk.config.gain == 9.0

    def test_set_updates_value_and_block(self):
        blk = _DummyBlock(value=0.0)
        p = Parameter(name="p", value=1.0, block=blk, attribute="value")
        p.set(5.0)
        assert p.value == 5.0
        assert blk.value == 5.0

    def test_value_setter(self):
        p = Parameter(name="x", value=1.0)
        p.value = 7.0
        assert p.value == 7.0

    def test_block_without_attribute_raises(self):
        blk = _DummyBlock()
        with pytest.raises(ValueError, match="attribute must be provided"):
            Parameter(name="bad", block=blk)

    def test_repr_free(self):
        p = Parameter(name="x", value=1.0)
        r = repr(p)
        assert "x" in r
        assert "bounds" in r

    def test_repr_block(self):
        blk = _DummyBlock()
        p = Parameter(name="p", value=1.0, block=blk, attribute="value")
        r = repr(p)
        assert "_DummyBlock" in r
        assert "value" in r


class TestBlockParameterFactory:

    def test_basic(self):
        blk = _DummyBlock(value=5.0)
        p = BlockParameter(blk, "value", value=3.0)
        assert p.name == "_DummyBlock.value"
        assert blk.value == 3.0

    def test_custom_name(self):
        blk = _DummyBlock()
        p = BlockParameter(blk, "value", name="my_param", value=1.0)
        assert p.name == "my_param"

    def test_with_transform(self):
        blk = _DummyBlock()
        p = BlockParameter(blk, "value", value=0.0, transform=lambda x: x + 10)
        assert blk.value == 10.0


class TestFreeParameterFactory:

    def test_basic(self):
        p = FreeParameter("alpha", value=2.5, bounds=(0, 5))
        assert p.name == "alpha"
        assert p.value == 2.5
        assert p.is_free_parameter


class TestSharedBlockParameter:

    def test_set_applies_to_all_targets(self):
        blk1 = _DummyBlock(value=0.0)
        blk2 = _DummyBlock(value=0.0)
        p = SharedBlockParameter(
            name="shared",
            targets=[blk1, blk2],
            attribute="value",
            value=7.0,
        )
        assert blk1.value == 7.0
        assert blk2.value == 7.0

        p.set(3.0)
        assert blk1.value == 3.0
        assert blk2.value == 3.0

    def test_with_transform(self):
        blk1 = _DummyBlock()
        blk2 = _DummyBlock()
        p = SharedBlockParameter(
            name="shared_log",
            targets=[blk1, blk2],
            attribute="value",
            value=2.0,
            transform=lambda x: x * 10,
        )
        assert blk1.value == 20.0
        assert blk2.value == 20.0

    def test_empty_targets_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SharedBlockParameter(name="bad", targets=[], attribute="value")


# ═══════════════════════════════════════════════════════════════════════════
# ScopeSignal tests
# ═══════════════════════════════════════════════════════════════════════════

class TestScopeSignal:

    def test_read_1d(self):
        scope = _DummyScope(t=np.array([0, 1, 2]), y=np.array([10, 20, 30]))
        sig = ScopeSignal(scope=scope, port=0)
        t, y = sig.read()
        np.testing.assert_array_equal(t, [0, 1, 2])
        np.testing.assert_array_equal(y, [10, 20, 30])

    def test_read_2d_selects_port(self):
        scope = _DummyScope(
            t=np.array([0, 1, 2]),
            y=np.array([[10, 20, 30], [40, 50, 60]]),
        )
        sig = ScopeSignal(scope=scope, port=1)
        t, y = sig.read()
        np.testing.assert_array_equal(y, [40, 50, 60])

    def test_read_none_scope_raises(self):
        sig = ScopeSignal(scope=None, port=0)
        with pytest.raises(ValueError, match="scope is None"):
            sig.read()


# ═══════════════════════════════════════════════════════════════════════════
# SimRunner tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSimRunner:

    def test_run_resets_and_runs(self):
        sim = _DummySim()
        runner = SimRunner(sim=sim, output=None, duration=5.0)
        runner.run()
        assert sim._reset_count == 1
        assert sim._ran

    def test_post_reset_hooks(self):
        sim = _DummySim()
        called = []
        runner = SimRunner(sim=sim, output=None, duration=1.0)
        runner.after_reset(lambda: called.append("hook1"))
        runner.after_reset(lambda: called.append("hook2"))
        runner.run()
        assert called == ["hook1", "hook2"]

    def test_pre_run_hook(self):
        sim = _DummySim()
        called = []
        runner = SimRunner(sim=sim, output=None, duration=1.0)
        runner.before_run(lambda: called.append("pre"))
        runner.run()
        assert "pre" in called

    def test_suppress_reset_log(self):
        sim = _DummySim()
        sim.log = True
        runner = SimRunner(sim=sim, output=None, duration=1.0, suppress_reset_log=True)
        runner.run()
        # log should be restored after run
        assert sim.log is True


# ═══════════════════════════════════════════════════════════════════════════
# block_param_to_var / free_param_to_var tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParamFactoryFunctions:

    def test_block_param_to_var_defaults(self):
        blk = _DummyBlock(value=42.0)
        p = block_param_to_var(blk, "value")
        assert p.value == 42.0
        assert "_DummyBlock.value" in p.name

    def test_block_param_to_var_with_param_id(self):
        blk = _DummyBlock(value=1.0)
        p = block_param_to_var(blk, "value", param_id="my_block")
        assert p.name == "my_block.value"

    def test_block_param_to_var_with_transform(self):
        blk = _DummyBlock(value=0.0)
        p = block_param_to_var(blk, "value", value=2.0, transform=np.exp)
        assert p() == pytest.approx(np.exp(2.0))

    def test_block_param_to_var_missing_attr(self):
        blk = _DummyBlock()
        with pytest.raises(AttributeError):
            block_param_to_var(blk, "nonexistent")

    def test_free_param_to_var(self):
        p = free_param_to_var("k", value=3.0, bounds=(0, 10))
        assert p.value == 3.0
        assert p.is_free_parameter

    def test_free_param_to_var_no_value_raises(self):
        with pytest.raises(ValueError, match="Initial value"):
            free_param_to_var("k")


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator tests
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterEstimatorInit:

    def test_empty_init(self):
        est = ParameterEstimator()
        assert est.parameters == []
        assert est.experiments == []

    def test_init_with_parameters(self):
        p = Parameter(name="x", value=1.0)
        est = ParameterEstimator(parameters=[p])
        assert len(est.parameters) == 1
        assert est.parameters[0] is p

    def test_init_with_simulator(self):
        scope = _DummyScope()
        blk = _DummyBlock()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        assert len(est.experiments) == 1


class TestParameterEstimatorExperiments:

    def _make_sim(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        return _DummySim(blocks=[blk, scope]), scope, blk

    def test_add_experiment(self):
        sim, scope, blk = self._make_sim()
        est = ParameterEstimator(simulator=sim)
        idx = est.add_experiment(sim, copy_sim=True)
        assert idx == 1
        assert len(est.experiments) == 2

    def test_ensure_experiment_creates_copies(self):
        sim, scope, blk = self._make_sim()
        est = ParameterEstimator(simulator=sim)
        est._ensure_experiment(2)
        assert len(est.experiments) == 3

    def test_ensure_experiment_negative_raises(self):
        est = ParameterEstimator()
        with pytest.raises(IndexError):
            est._ensure_experiment(-1)


class TestParameterEstimatorParameters:

    def _make_est(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        return est, sim, scope, blk

    def test_add_block_parameter(self):
        est, sim, scope, blk = self._make_est()
        est.add_block_parameter(blk, "value", value=5.0, param_id="test")
        assert len(est.global_parameters) == 1
        assert est.global_parameters[0].value == 5.0

    def test_add_block_parameter_with_transform(self):
        est, sim, scope, blk = self._make_est()
        est.add_block_parameter(blk, "value", value=2.0, transform=np.exp)
        p = est.global_parameters[0]
        assert p() == pytest.approx(np.exp(2.0))

    def test_add_global_block_parameter(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)
        est.add_global_block_parameter("_DummyBlock", "value", value=10.0)
        assert len(est.global_parameters) == 1

    def test_add_local_block_parameter(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)
        est.add_local_block_parameter(0, "_DummyBlock", "value", value=2.0)
        est.add_local_block_parameter(1, "_DummyBlock", "value", value=3.0)
        assert len(est.local_parameters[0]) == 1
        assert len(est.local_parameters[1]) == 1

    def test_parameters_property_order(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        est.add_global_block_parameter("_DummyBlock", "value", value=10.0, param_id="global")
        est.add_local_block_parameter(0, "_DummyBlock", "gain", value=2.0, param_id="local")
        est.add_local_block_parameter(1, "_DummyBlock", "gain", value=3.0, param_id="local")

        params = est.parameters
        assert len(params) == 3
        assert "global" in params[0].name
        assert "exp0" in params[1].name
        assert "exp1" in params[2].name


class TestParameterEstimatorTimeseries:

    def test_add_timeseries_with_signal(self):
        scope = _DummyScope()
        blk = _DummyBlock()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1, 2]), data=np.array([0, 1, 2]))
        est.add_timeseries(ts, signal=scope[0], sigma=1.0)

        assert len(est.experiments[0].measurements) == 1
        assert len(est.experiments[0].outputs) == 1
        assert est.experiments[0].sigma[0] == 1.0

    def test_add_timeseries_with_scope_port(self):
        scope = _DummyScope()
        blk = _DummyBlock()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        est.add_timeseries(ts, scope=scope, port=0)

        assert len(est.experiments[0].measurements) == 1

    def test_add_timeseries_both_signal_and_scope_raises(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        with pytest.raises(ValueError, match="not both"):
            est.add_timeseries(ts, signal=scope[0], scope=scope, port=0)

    def test_add_timeseries_no_scope_raises(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        with pytest.raises(ValueError, match="must pass"):
            est.add_timeseries(ts)

    def test_add_timeseries_wrong_type_raises(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)

        with pytest.raises(TypeError, match="TimeSeriesData"):
            est.add_timeseries("not a timeseries", scope=scope, port=0)

    def test_duration_derived_from_measurements(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 5, 10]), data=np.array([0, 1, 2]))
        est.add_timeseries(ts, scope=scope, port=0)

        assert est.duration == 10.0


class TestParameterEstimatorApply:

    def test_apply_sets_parameters(self):
        blk = _DummyBlock(value=0.0)
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=1.0)

        est.apply(np.array([5.0]))
        assert blk.value == 5.0

    def test_apply_wrong_length_raises(self):
        blk = _DummyBlock()
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=1.0)

        with pytest.raises(ValueError, match="Expected x of length"):
            est.apply(np.array([1.0, 2.0]))

    def test_apply_global_and_local(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        est.add_global_block_parameter("_DummyBlock", "value", value=1.0, param_id="g")
        est.add_local_block_parameter(0, "_DummyBlock", "gain", value=1.0, param_id="l")
        est.add_local_block_parameter(1, "_DummyBlock", "gain", value=1.0, param_id="l")

        # x = [global_value, local_gain_exp0, local_gain_exp1]
        est.apply(np.array([10.0, 20.0, 30.0]))

        assert est.global_parameters[0].value == 10.0
        assert est.local_parameters[0][0].value == 20.0
        assert est.local_parameters[1][0].value == 30.0


class TestParameterEstimatorSimulate:

    def test_simulate_runs_experiment(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1, 2]), data=np.array([0, 1, 2]))
        est.add_timeseries(ts, scope=scope, port=0)

        t, y = est.simulate(np.array([]))
        assert len(t) > 0
        assert len(y) > 0

    def test_simulate_no_experiments_raises(self):
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.simulate(np.array([]))

    def test_simulate_bad_experiment_raises(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)
        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        est.add_timeseries(ts, scope=scope, port=0)

        with pytest.raises(IndexError):
            est.simulate(np.array([]), experiment=5)


class TestParameterEstimatorResiduals:

    def test_residuals_shape(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(
            time=np.array([0.0, 0.5, 1.0]),
            data=np.array([0.0, 0.5, 1.0]),
        )
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)

        r = est.residuals(np.array([1.0]))
        assert r.shape == (3,)

    def test_residuals_no_experiments_raises(self):
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.residuals(np.array([]))

    def test_residuals_multi_experiment(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        ts1 = TimeSeriesData(time=np.array([0.0, 0.5, 1.0]), data=np.array([0, 0.5, 1.0]))
        ts2 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1.0]))

        est.add_global_block_parameter("_DummyBlock", "value", value=1.0, param_id="g")
        est.add_timeseries(ts1, scope=scope, port=0, experiment=0)
        est.add_timeseries(ts2, scope=scope, port=0, experiment=1)

        r = est.residuals(np.array([1.0]))
        # 3 points from exp0 + 2 points from exp1
        assert r.shape == (5,)

    def test_residuals_empty_measurements(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)
        # No measurements added
        r = est.residuals(np.array([]))
        assert r.shape == (0,)


class TestParameterEstimatorDisplay:

    def test_display_runs(self, capsys):
        blk = _DummyBlock(value=1.0)
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=3.0, param_id="test")
        est.display()

        captured = capsys.readouterr()
        assert "Parameter Estimation Results" in captured.out
        assert "Global parameters" in captured.out
        assert "test.value" in captured.out

    def test_display_with_transform(self, capsys):
        blk = _DummyBlock(value=0.0)
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=2.0, transform=np.exp)
        est.display()

        captured = capsys.readouterr()
        assert "->" in captured.out

    def test_display_local_params(self, capsys):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)
        est.add_local_block_parameter(0, "_DummyBlock", "value", value=2.0)
        est.display()

        captured = capsys.readouterr()
        assert "Local parameters (experiment 0)" in captured.out


class TestParameterEstimatorFit:

    def test_fit_least_squares(self):
        """Fit a simple y = gain * t model to data with known slope."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        # Target: gain=2.0
        t_meas = np.array([0.0, 0.5, 1.0])
        y_meas = 2.0 * t_meas
        ts = TimeSeriesData(time=t_meas, data=y_meas)

        est.add_block_parameter(blk, "value", value=1.0, bounds=(0.1, 10.0))
        est.add_timeseries(ts, scope=scope, port=0)

        result = est.fit(max_nfev=50)
        assert isinstance(result, EstimatorResult)
        assert result.x[0] == pytest.approx(2.0, abs=0.2)

    def test_fit_with_constraints_least_squares_raises(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)

        with pytest.raises(ValueError, match="does not support general constraints"):
            est.fit(constraints=[{"type": "eq", "fun": lambda x: x[0] - 1}])


class TestParameterEstimatorPlotFit:

    def test_plot_fit_returns_fig_axes(self):
        pytest.importorskip("matplotlib")

        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 0.5, 1.0]), data=np.array([0, 0.5, 1.0]))
        est.add_timeseries(ts, scope=scope, port=0)

        import matplotlib
        matplotlib.use("Agg")

        fig, axes = est.plot_fit(np.array([]))
        assert fig is not None
        assert len(axes) == 1

    def test_plot_fit_overlay(self):
        pytest.importorskip("matplotlib")

        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        ts1 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        ts2 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 2]))
        est.add_timeseries(ts1, scope=scope, port=0, experiment=0)
        est.add_timeseries(ts2, scope=scope, port=0, experiment=1)

        import matplotlib
        matplotlib.use("Agg")

        fig, axes = est.plot_fit(np.array([]), overlay=True)
        assert len(axes) == 1

    def test_plot_fit_no_experiments_raises(self):
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.plot_fit(np.array([]))


class TestScopePortFromSignal:

    def test_port_reference(self):
        scope = _DummyScope()
        ref = scope[0]
        s, p = ParameterEstimator._scope_port_from_signal(ref)
        assert s is scope
        assert p == 0

    def test_tuple(self):
        scope = _DummyScope()
        s, p = ParameterEstimator._scope_port_from_signal((scope, 1))
        assert s is scope
        assert p == 1

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported signal type"):
            ParameterEstimator._scope_port_from_signal("bad")


class TestBlockOccurrence:

    def test_finds_block(self):
        s1 = _DummyScope()
        s2 = _DummyScope()
        sim = _DummySim(blocks=[_DummyBlock(), s1, _DummyBlock(), s2])
        tname, occ = ParameterEstimator._block_occurrence(sim, s2)
        assert tname == "_DummyScope"
        assert occ == 1

    def test_not_found_raises(self):
        sim = _DummySim(blocks=[_DummyBlock()])
        with pytest.raises(ValueError, match="block not found"):
            ParameterEstimator._block_occurrence(sim, _DummyScope())

    def test_find_by_occurrence(self):
        s1 = _DummyScope()
        s2 = _DummyScope()
        sim = _DummySim(blocks=[s1, _DummyBlock(), s2])
        found = ParameterEstimator._find_block_by_occurrence(sim, "_DummyScope", 1)
        assert found is s2

    def test_find_by_occurrence_not_found_raises(self):
        sim = _DummySim(blocks=[_DummyBlock()])
        with pytest.raises(ValueError, match="No block"):
            ParameterEstimator._find_block_by_occurrence(sim, "_DummyScope", 0)


class TestResolveOutput:

    def test_resolve_with_scope_set(self):
        scope = _DummyScope()
        sig = ScopeSignal(scope=scope, port=0)
        sim = _DummySim(blocks=[scope])
        exp = Experiment(runner=SimRunner(sim=sim, output=None, duration=1.0))
        est = ParameterEstimator()

        resolved = est._resolve_output(exp, sig)
        assert resolved.scope is scope

    def test_resolve_by_occurrence(self):
        s1 = _DummyScope()
        s2 = _DummyScope()
        sim = _DummySim(blocks=[s1, _DummyBlock(), s2])
        exp = Experiment(runner=SimRunner(sim=sim, output=None, duration=1.0))
        est = ParameterEstimator()

        sig = ScopeSignal(scope=None, port=0, block_type="_DummyScope", occurrence=1)
        resolved = est._resolve_output(exp, sig)
        assert resolved.scope is s2

    def test_resolve_no_sim_raises(self):
        est = ParameterEstimator()
        sig = ScopeSignal(scope=None, port=0, block_type="X", occurrence=0)
        exp = Experiment(runner="not_a_runner")
        with pytest.raises(ValueError, match="SimRunner"):
            est._resolve_output(exp, sig)


class TestEstimatorResult:

    def test_fields(self):
        r = EstimatorResult(
            x=np.array([1.0]),
            cost=0.5,
            nfev=10,
            success=True,
            message="ok",
        )
        assert r.success
        assert r.cost == 0.5
        assert r.nfev == 10
        assert r.message == "ok"
