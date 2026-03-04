#########################################################################################
##
##                                     TESTS FOR
##            'opt/parameter_estimator.py' and 'opt/timeseries_data.py'
##
##                                  Kevin McBride 2026
##
#########################################################################################

# IMPORTS ===============================================================================

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
    block_param_to_var,   # internal helper — tested directly
    free_param_to_var,    # internal helper — tested directly
)
from pathsim.utils.timeseries_data import TimeSeriesData


# ═══════════════════════════════════════════════════════════════════════════
# Test helpers / dummy classes
# ═══════════════════════════════════════════════════════════════════════════

class _DummyBlock:
    """Minimal block-like object for unit tests."""

    def __init__(self, value=1.0, gain=2.0):
        self.value = value
        self.gain = gain

    def update(self, t):
        pass


class _NestedBlock:
    """Block with dotted attribute path."""

    class Config:
        def __init__(self):
            self.gain = 5.0

    def __init__(self):
        self.config = self.Config()

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
        for b in self.blocks:
            if isinstance(b, _DummyScope):
                t = np.linspace(0, duration, 50)
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
        p = Parameter(
            name="blk.value", value=2.0, block=blk, attribute="value",
            transform=lambda x: x ** 2,
        )
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
        sig_data = sig.read()
        t = sig_data.time
        y = sig_data.data
        np.testing.assert_array_equal(t, [0, 1, 2])
        np.testing.assert_array_equal(y, [10, 20, 30])

    def test_read_2d_selects_port(self):
        # (n_ports, n_time) layout
        scope = _DummyScope(
            t=np.array([0, 1, 2]),
            y=np.array([[10, 20, 30], [40, 50, 60]]),
        )
        sig = ScopeSignal(scope=scope, port=1)
        sig_data = sig.read()
        t = sig_data.time
        y = sig_data.data
        np.testing.assert_array_equal(y, [40, 50, 60])

    def test_read_2d_time_first_layout(self):
        # (n_time, n_ports) layout — should be auto-transposed
        scope = _DummyScope(
            t=np.array([0, 1, 2]),
            y=np.array([[10, 40], [20, 50], [30, 60]]),  # shape (3, 2)
        )
        sig = ScopeSignal(scope=scope, port=1)
        sig_data = sig.read()
        t = sig_data.time
        y = sig_data.data
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
# ParameterEstimator — init
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

    def test_runners_property_in_sync(self):
        scope = _DummyScope()
        blk = _DummyBlock()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        assert len(est.runners) == 1
        est.add_experiment(sim, copy_sim=True)
        assert len(est.runners) == 2


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — experiments
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — parameter registration
# ═══════════════════════════════════════════════════════════════════════════

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

    def test_add_global_block_parameter_two_experiments(self):
        """Global block parameter targets matching blocks in both experiments."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)
        est.add_global_block_parameter("_DummyBlock", "value", value=5.0)
        assert len(est.global_parameters) == 1
        assert isinstance(est.global_parameters[0], SharedBlockParameter)

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

    def test_add_local_block_parameter_with_transform(self):
        """Local block parameters accept a transform argument."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_local_block_parameter(
            0, "_DummyBlock", "value", value=2.0, transform=np.exp
        )
        p = est.local_parameters[0][0]
        assert p() == pytest.approx(np.exp(2.0))
        assert blk.value == pytest.approx(np.exp(2.0))

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


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — time series
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — apply
# ═══════════════════════════════════════════════════════════════════════════

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

        est.apply(np.array([10.0, 20.0, 30.0]))

        assert est.global_parameters[0].value == 10.0
        assert est.local_parameters[0][0].value == 20.0
        assert est.local_parameters[1][0].value == 30.0

    def test_apply_empty_vector(self):
        """apply([]) should succeed when there are no parameters."""
        est = ParameterEstimator()
        est.apply(np.array([]))  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — simulate
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — residuals
# ═══════════════════════════════════════════════════════════════════════════

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
        r = est.residuals(np.array([]))
        assert r.shape == (0,)

    def test_residuals_sigma_scaling(self):
        """Non-unit sigma scales the residuals."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0.0, 0.5]))
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0, sigma=2.0)

        r_sigma1 = est.residuals(np.array([1.0]))
        # Manually add same ts with sigma=1
        est2 = ParameterEstimator(simulator=sim)
        est2.add_block_parameter(blk, "value", value=1.0)
        est2.add_timeseries(ts, scope=scope, port=0, sigma=1.0)
        r_sigma1_ref = est2.residuals(np.array([1.0]))

        np.testing.assert_allclose(r_sigma1, r_sigma1_ref / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — display
# ═══════════════════════════════════════════════════════════════════════════

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

    def test_display_shows_bounds(self, capsys):
        blk = _DummyBlock(value=1.0)
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=1.0, bounds=(0.0, 10.0))
        est.display()

        captured = capsys.readouterr()
        assert "[0" in captured.out
        assert "10" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — fit
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterEstimatorFit:

    def test_fit_least_squares(self):
        """Fit a simple y = gain * t model to data with known slope."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

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

    def test_fit_minimize_method(self):
        """Fit using scipy minimize (L-BFGS-B)."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        t_meas = np.array([0.0, 0.5, 1.0])
        y_meas = 3.0 * t_meas
        ts = TimeSeriesData(time=t_meas, data=y_meas)

        est.add_block_parameter(blk, "value", value=1.0, bounds=(0.1, 10.0))
        est.add_timeseries(ts, scope=scope, port=0)

        result = est.fit(max_nfev=100, method="L-BFGS-B")
        assert isinstance(result, EstimatorResult)
        assert result.x[0] == pytest.approx(3.0, abs=0.5)

    def test_fit_result_repr(self):
        r = EstimatorResult(
            x=np.array([1.0, 2.0]),
            cost=0.01,
            nfev=42,
            success=True,
            message="converged",
        )
        s = repr(r)
        assert "SUCCESS" in s
        assert "0.01" in s
        assert "42" in s


# ═══════════════════════════════════════════════════════════════════════════
# ParameterEstimator — plot_fit
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterEstimatorPlotFit:

    def test_plot_fit_returns_fig_axes(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")

        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 0.5, 1.0]), data=np.array([0, 0.5, 1.0]))
        est.add_timeseries(ts, scope=scope, port=0)

        fig, axes = est.plot_fit(np.array([]))
        assert fig is not None
        assert len(axes) == 1

    def test_plot_fit_overlay(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")

        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        ts1 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        ts2 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 2]))
        est.add_timeseries(ts1, scope=scope, port=0, experiment=0)
        est.add_timeseries(ts2, scope=scope, port=0, experiment=1)

        fig, axes = est.plot_fit(np.array([]), overlay=True)
        assert len(axes) == 1

    def test_plot_fit_no_experiments_raises(self):
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.plot_fit(np.array([]))

    def test_plot_fit_multiple_datasets_per_experiment(self):
        """plot_fit should iterate all datasets, not just output 0."""
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")

        scope1 = _DummyScope()
        scope2 = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope1, scope2])
        est = ParameterEstimator(simulator=sim)

        ts1 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        ts2 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 2]))
        est.add_timeseries(ts1, scope=scope1, port=0, experiment=0)
        est.add_timeseries(ts2, scope=scope2, port=0, experiment=0)

        fig, axes = est.plot_fit(np.array([]))
        assert fig is not None


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# EstimatorResult
# ═══════════════════════════════════════════════════════════════════════════

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

    def test_repr_success(self):
        r = EstimatorResult(
            x=np.array([1.0]), cost=0.0, nfev=5, success=True, message="ok"
        )
        assert "SUCCESS" in repr(r)

    def test_repr_failure(self):
        r = EstimatorResult(
            x=np.array([1.0]), cost=1.0, nfev=80, success=False, message="maxiter"
        )
        assert "FAILED" in repr(r)


# ═══════════════════════════════════════════════════════════════════════════
# Stress tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStress:

    def test_multi_experiment_convergence(self):
        """Two experiments, one global + two local parameters converge correctly."""
        scope1 = _DummyScope()
        blk1 = _DummyBlock(value=1.0)
        sim1 = _DummySim(blocks=[blk1, scope1])

        scope2 = _DummyScope()
        blk2 = _DummyBlock(value=1.0)
        sim2 = _DummySim(blocks=[blk2, scope2])

        # Both sims use the same gain (global) but different offsets are
        # handled by the gain itself in this simple model.
        est = ParameterEstimator(simulator=sim1)
        est.add_experiment(sim2)

        # Global: shared gain ~ 3.0
        est.add_global_block_parameter("_DummyBlock", "value", value=1.0, param_id="gain")

        t = np.linspace(0, 1, 11)
        ts1 = TimeSeriesData(time=t, data=3.0 * t, name="exp0")
        ts2 = TimeSeriesData(time=t, data=3.0 * t, name="exp1")
        est.add_timeseries(ts1, scope=scope1, port=0, sigma=1.0, experiment=0)
        est.add_timeseries(ts2, scope=scope2, port=0, sigma=1.0, experiment=1)

        result = est.fit(max_nfev=80)
        assert result.x[0] == pytest.approx(3.0, abs=0.3)

    def test_transform_applied_during_fit(self):
        """Transform is applied correctly throughout the optimization."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        # Fit in log space: exp(x) = 4.0 → x = ln(4) ≈ 1.386
        t = np.linspace(0, 1, 11)
        ts = TimeSeriesData(time=t, data=4.0 * t)

        est.add_block_parameter(
            blk, "value", value=1.0, bounds=(-5, 5), transform=np.exp
        )
        est.add_timeseries(ts, scope=scope, port=0)

        result = est.fit(max_nfev=100)
        assert blk.value == pytest.approx(4.0, abs=0.5)

    def test_parameters_at_bounds(self):
        """Residuals remain finite even when parameters are at their bounds."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 0.5, 1.0]), data=np.array([0, 0.5, 1]))
        est.add_block_parameter(blk, "value", value=0.1, bounds=(0.1, 10.0))
        est.add_timeseries(ts, scope=scope, port=0)

        r = est.residuals(np.array([0.1]))  # at lower bound
        assert np.all(np.isfinite(r))

        r = est.residuals(np.array([10.0]))  # at upper bound
        assert np.all(np.isfinite(r))

    def test_free_parameter_in_estimator(self):
        """Free parameters (not block-bound) are handled correctly."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        # Free parameter: just a scalar, not bound to any block
        scale = Parameter("scale", value=2.0, bounds=(0.1, 10.0))
        est.add_parameters([scale])

        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0.0, 1.0]))
        est.add_timeseries(ts, scope=scope, port=0)

        # apply() should set scale.value but not crash (no block)
        est.apply(np.array([5.0]))
        assert scale.value == 5.0

    def test_missing_outputs_raises(self):
        """Mismatched measurements/outputs raises a clear error."""
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)

        # Manually break the invariant to test the guard
        ts = TimeSeriesData(time=np.array([0, 1]), data=np.array([0, 1]))
        est.add_timeseries(ts, scope=scope, port=0)
        # Add an extra output without a corresponding measurement
        est.experiments[0].outputs.append(
            ScopeSignal(scope=scope, port=0)
        )

        with pytest.raises(ValueError, match="output mapping"):
            est.residuals(np.array([]))

    def test_add_global_no_experiments_raises(self):
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.add_global_block_parameter("_DummyBlock", "value")

    def test_add_local_missing_block_raises(self):
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)
        with pytest.raises(ValueError, match="no block of type"):
            est.add_local_block_parameter(0, "NonExistentBlock", "value")

    def test_add_local_missing_attr_raises(self):
        blk = _DummyBlock()
        scope = _DummyScope()
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        with pytest.raises(AttributeError):
            est.add_local_block_parameter(0, "_DummyBlock", "nonexistent_attr")


# ═══════════════════════════════════════════════════════════════════════════
# New robustness tests (from agent review fixes)
# ═══════════════════════════════════════════════════════════════════════════

class TestParameterBoundsValidation:

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="lower bound"):
            Parameter(name="x", value=1.0, bounds=(5.0, 2.0))

    def test_value_below_lower_bound_warns(self):
        with pytest.warns(UserWarning, match="initial value"):
            Parameter(name="x", value=-1.0, bounds=(0.0, 10.0))

    def test_value_above_upper_bound_warns(self):
        with pytest.warns(UserWarning, match="initial value"):
            Parameter(name="x", value=20.0, bounds=(0.0, 10.0))

    def test_value_at_lower_bound_no_warn(self):
        # Exactly at bound — no warning
        p = Parameter(name="x", value=0.0, bounds=(0.0, 10.0))
        assert p.value == 0.0

    def test_value_at_upper_bound_no_warn(self):
        p = Parameter(name="x", value=10.0, bounds=(0.0, 10.0))
        assert p.value == 10.0

    def test_infinite_bounds_no_validation(self):
        # Infinite bounds should never warn or raise
        p = Parameter(name="x", value=999.0)
        assert p.value == 999.0


class TestScopeSignalExtractRobustness:

    def test_square_array_raises(self):
        """A square 2D scope output is ambiguous and should raise."""
        scope = _DummyScope(
            t=np.linspace(0, 1, 4),
            y=np.ones((4, 4)),  # square — ambiguous
        )
        sig = ScopeSignal(scope=scope, port=0)
        with pytest.raises(ValueError, match="ambiguous"):
            sig.read()

    def test_port_out_of_range_raises(self):
        """Requesting a port beyond the scope's n_ports raises IndexError."""
        scope = _DummyScope(
            t=np.array([0.0, 1.0, 2.0]),
            y=np.array([[1.0, 2.0, 3.0]]),  # 1 port, (1, 3)
        )
        sig = ScopeSignal(scope=scope, port=5)
        with pytest.raises(IndexError, match="port=5"):
            sig.read()

    def test_non_square_time_first_auto_transpose(self):
        """(n_time, n_ports) layout is transposed when unambiguous."""
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])  # (3, 2) time-first
        scope = _DummyScope(t=t, y=y)
        sig = ScopeSignal(scope=scope, port=0)
        sig_data = sig.read()
        y_out = sig_data.data
        np.testing.assert_array_equal(y_out, [10.0, 20.0, 30.0])


class TestPreFitValidation:

    def test_fit_no_parameters_raises(self):
        """fit() with no parameters should raise before running."""
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)
        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0.0, 1.0]))
        est.add_timeseries(ts, scope=scope, port=0)

        with pytest.raises(ValueError, match="No parameters"):
            est.fit()

    def test_fit_no_measurements_raises(self):
        """fit() with no measurements should raise before running."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(blk, "value", value=1.0)

        with pytest.raises(ValueError, match="No measurements"):
            est.fit()

    def test_fit_no_experiments_raises(self):
        """fit() on a completely empty estimator should raise."""
        est = ParameterEstimator()
        with pytest.raises(ValueError, match="No experiments"):
            est.fit()


class TestCopySimWarning:

    def test_shared_sim_warns(self):
        """Registering the same sim twice without copy_sim=True should warn."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        with pytest.warns(UserWarning, match="copy_sim"):
            est.add_experiment(sim)  # same object, copy_sim=False

    def test_different_sim_no_warn(self):
        """Different sim objects should not warn even without copy_sim."""
        scope1 = _DummyScope()
        blk1 = _DummyBlock(value=1.0)
        sim1 = _DummySim(blocks=[blk1, scope1])

        scope2 = _DummyScope()
        blk2 = _DummyBlock(value=1.0)
        sim2 = _DummySim(blocks=[blk2, scope2])

        est = ParameterEstimator(simulator=sim1)
        # Should not warn — different objects
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            est.add_experiment(sim2)  # no warning expected


# ═══════════════════════════════════════════════════════════════════════════
# Caching tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCaching:

    def _make_est(self):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        ts = TimeSeriesData(time=np.array([0.0, 0.5, 1.0]), data=np.array([0, 0.5, 1]))
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)
        return est, blk, sim

    def test_residuals_cached_on_repeat_call(self):
        """Calling residuals() twice with the same x skips re-running the sim."""
        est, blk, sim = self._make_est()
        x = np.array([1.0])
        r1 = est.residuals(x)
        sim._ran = False           # reset the flag
        r2 = est.residuals(x)     # should use cache — sim should NOT run again
        assert not sim._ran, "simulation re-ran despite identical x"
        np.testing.assert_array_equal(r1, r2)

    def test_residuals_recomputed_on_different_x(self):
        """Calling residuals() with a new x must recompute."""
        est, blk, sim = self._make_est()
        r1 = est.residuals(np.array([1.0]))
        sim._ran = False
        r2 = est.residuals(np.array([2.0]))
        assert sim._ran, "simulation did not re-run for different x"
        assert not np.array_equal(r1, r2)

    def test_cache_invalidated_on_add_parameter(self):
        """Adding a parameter after first residuals call forces recomputation."""
        est, blk, sim = self._make_est()
        est.residuals(np.array([1.0]))
        assert est._params_cache is not None

        est.add_block_parameter(blk, "gain", value=2.0)
        assert est._params_cache is None, "params cache not cleared after add_block_parameter"

    def test_cache_invalidated_on_add_timeseries(self):
        """Adding a new measurement invalidates the residuals cache."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)
        est.residuals(np.array([1.0]))

        ts2 = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 2]))
        est.add_timeseries(ts2, scope=scope, port=0)
        assert est._cached_x is None, "residuals cache not cleared after add_timeseries"

    def test_parameters_property_cached(self):
        """parameters property returns the same list object on repeat access."""
        est, blk, sim = self._make_est()
        p1 = est.parameters
        p2 = est.parameters
        assert p1 is p2, "parameters property should return cached list"

    def test_parameters_cache_invalidated_by_add_parameters(self):
        est, blk, sim = self._make_est()
        _ = est.parameters               # populate cache
        est.add_parameters([Parameter("free", value=1.0)])
        assert est._params_cache is None


# ═══════════════════════════════════════════════════════════════════════════
# NaN / Inf handling tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNanInfHandling:

    def test_residuals_nan_in_measurement_propagates(self):
        """NaN in measurement data propagates to residuals (not silently dropped)."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        t = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, np.nan, 1.0])
        ts = TimeSeriesData.__new__(TimeSeriesData)  # bypass validation
        ts.time = t
        ts.data = y
        ts.name = "test"
        ts.time_info = {"time_range": {"start": 0.0, "end": 1.0}, "units": "s"}

        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)

        r = est.residuals(np.array([1.0]))
        assert np.any(np.isnan(r)), "NaN in measurement should propagate to residuals"

    def test_residuals_finite_for_normal_data(self):
        """Residuals are fully finite for clean data."""
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
        assert np.all(np.isfinite(r))


# ═══════════════════════════════════════════════════════════════════════════
# TimeSeriesData.plot() ax= parameter
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeSeriesDataPlot:

    def test_plot_returns_ax(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = TimeSeriesData(
            time=np.array([0.0, 1.0, 2.0]),
            data=np.array([0.0, 1.0, 2.0]),
        )
        ax = ts.plot()
        assert ax is not None

    def test_plot_reuses_existing_ax(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ts = TimeSeriesData(
            time=np.array([0.0, 1.0, 2.0]),
            data=np.array([0.0, 1.0, 2.0]),
        )
        returned = ts.plot(ax=ax)
        assert returned is ax


# ═══════════════════════════════════════════════════════════════════════════
# SharedBlockParameter — composition-safe init
# ═══════════════════════════════════════════════════════════════════════════

class TestSharedBlockParameterInit:

    def test_init_sets_all_targets(self):
        blk1 = _DummyBlock(value=0.0)
        blk2 = _DummyBlock(value=0.0)
        p = SharedBlockParameter("g", [blk1, blk2], "value", value=3.0)
        assert blk1.value == 3.0
        assert blk2.value == 3.0
        assert p.value == 3.0

    def test_init_order_independent(self):
        """set() is only called after self.targets is ready — no AttributeError."""
        blk1 = _DummyBlock(value=0.0)
        blk2 = _DummyBlock(value=0.0)
        # This would fail with the old super().__init__() ordering if targets
        # was not assigned before set() was dispatched.
        p = SharedBlockParameter("g", [blk1, blk2], "value", value=5.0)
        assert p() == 5.0

    def test_inverted_bounds_raises(self):
        blk = _DummyBlock()
        with pytest.raises(ValueError, match="lower bound"):
            SharedBlockParameter("g", [blk], "value", bounds=(5.0, 1.0))

    def test_value_out_of_bounds_warns(self):
        blk = _DummyBlock()
        with pytest.warns(UserWarning, match="initial value"):
            SharedBlockParameter("g", [blk], "value", value=20.0, bounds=(0, 10))

    def test_is_block_parameter(self):
        blk = _DummyBlock()
        p = SharedBlockParameter("g", [blk], "value", value=1.0)
        assert p.is_block_parameter
        assert not p.is_free_parameter


# ═══════════════════════════════════════════════════════════════════════════
# Constructor defaults forwarding in add_experiment()
# ═══════════════════════════════════════════════════════════════════════════

class TestConstructorDefaultsForwarding:

    def test_adaptive_inherited_by_add_experiment(self):
        """add_experiment() without explicit adaptive= inherits the constructor value."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim, adaptive=True)
        est.add_experiment(sim, copy_sim=True)

        # Both runners should have adaptive=True
        for runner in est.runners:
            assert runner.adaptive is True

    def test_explicit_adaptive_overrides_default(self):
        """Explicitly passing adaptive=False overrides the constructor default."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim, adaptive=True)
        est.add_experiment(sim, copy_sim=True, adaptive=False)

        assert est.runners[0].adaptive is True   # constructor default
        assert est.runners[1].adaptive is False  # explicit override

    def test_pre_run_inherited(self):
        """pre_run callable is inherited by subsequent add_experiment() calls."""
        hook_log = []
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])

        hook = lambda: hook_log.append("ran")
        est = ParameterEstimator(simulator=sim, pre_run=hook)
        est.add_experiment(sim, copy_sim=True)

        # Both runners should have the same hook
        assert est.runners[0].pre_run is hook
        assert est.runners[1].pre_run is hook


# ═══════════════════════════════════════════════════════════════════════════
# Deferred duration update
# ═══════════════════════════════════════════════════════════════════════════

class TestDeferredDurationUpdate:

    def test_duration_updated_before_fit(self):
        """Runner duration is correct by the time fit() evaluates residuals."""
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 5.0, 10.0]), data=np.array([0, 5, 10]))
        est.add_block_parameter(blk, "value", value=1.0)
        est.add_timeseries(ts, scope=scope, port=0)

        # Duration should be stale (dirty) until residuals/fit forces update
        assert est._duration_dirty

        est.residuals(np.array([1.0]))

        # After residuals(), duration should be current
        assert not est._duration_dirty
        assert est.runners[0].duration == pytest.approx(10.0)

    def test_duration_not_updated_without_measurements(self):
        """_duration_dirty stays False if no measurements have been added yet."""
        scope = _DummyScope()
        sim = _DummySim(blocks=[scope])
        est = ParameterEstimator(simulator=sim)
        # No measurements — dirty should be True (experiment was added)
        assert est._duration_dirty
        est._ensure_duration_current()
        assert not est._duration_dirty   # resolved: no measurements → duration stays 0


# ═══════════════════════════════════════════════════════════════════════════
# Constraint-based fitting (minimize path)
# ═══════════════════════════════════════════════════════════════════════════

class TestConstraintFitting:

    def _make_est(self, target_gain=2.0):
        scope = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, scope])
        est = ParameterEstimator(simulator=sim)
        t = np.linspace(0, 1, 11)
        ts = TimeSeriesData(time=t, data=target_gain * t)
        est.add_block_parameter(blk, "value", value=1.0, bounds=(0.1, 10.0))
        est.add_timeseries(ts, scope=scope, port=0)
        return est, blk

    def test_slsqp_unconstrained(self):
        est, blk = self._make_est(target_gain=2.0)
        result = est.fit(method="SLSQP", max_nfev=100)
        assert isinstance(result, EstimatorResult)
        assert result.x[0] == pytest.approx(2.0, abs=0.3)

    def test_slsqp_with_inequality_constraint(self):
        """SLSQP respects an inequality constraint that clips the optimal value."""
        est, blk = self._make_est(target_gain=2.0)
        # Constrain gain >= 2.5 — optimizer should land at ~2.5, not 2.0
        constraints = [{"type": "ineq", "fun": lambda x: x[0] - 2.5}]
        result = est.fit(method="SLSQP", max_nfev=100, constraints=constraints)
        assert result.x[0] >= 2.5 - 1e-4

    def test_constraints_rejected_for_least_squares(self):
        est, _ = self._make_est()
        with pytest.raises(ValueError, match="does not support general constraints"):
            est.fit(constraints=[{"type": "eq", "fun": lambda x: x[0] - 1}])

    def test_cobyla_unconstrained(self):
        est, blk = self._make_est(target_gain=3.0)
        result = est.fit(method="COBYLA", max_nfev=200)
        assert isinstance(result, EstimatorResult)

    def test_constraint_method_on_unsupported_method_raises(self):
        est, _ = self._make_est()
        with pytest.raises(ValueError, match="does not support general constraints"):
            est.fit(method="L-BFGS-B",
                    constraints=[{"type": "ineq", "fun": lambda x: x[0]}])


# ═══════════════════════════════════════════════════════════════════════════
# Multi-experiment scope resolution (structural assumption documented)
# ═══════════════════════════════════════════════════════════════════════════

class TestScopeResolution:

    def test_same_structure_resolves_correctly(self):
        """Deep-copied sims with identical block order resolve to the right scope."""
        s1 = _DummyScope()
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, s1])
        est = ParameterEstimator(simulator=sim)
        est.add_experiment(sim, copy_sim=True)

        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        est.add_global_block_parameter("_DummyBlock", "value", value=1.0)
        est.add_timeseries(ts, signal=s1[0], sigma=1.0, experiment=0)
        est.add_timeseries(ts, signal=s1[0], sigma=1.0, experiment=1)

        # Both experiments should resolve scopes without error
        r = est.residuals(np.array([1.0]))
        assert r.shape == (4,)   # 2 points × 2 experiments

    def test_scope_not_in_sim_raises(self):
        """Resolving a scope that doesn't exist in the experiment sim raises."""
        s_external = _DummyScope()   # not in any sim
        blk = _DummyBlock(value=1.0)
        sim = _DummySim(blocks=[blk, _DummyScope()])
        est = ParameterEstimator(simulator=sim)

        ts = TimeSeriesData(time=np.array([0.0, 1.0]), data=np.array([0, 1]))
        est.add_block_parameter(blk, "value", value=1.0)
        # Force a direct scope reference (not resolved via occurrence)
        sig = ScopeSignal(scope=None, port=0, block_type="_DummyScope", occurrence=99)
        est.experiments[0].measurements.append(ts)
        est.experiments[0].outputs.append(sig)
        est.experiments[0].sigma.append(1.0)

        with pytest.raises(ValueError, match="No block"):
            est.residuals(np.array([1.0]))
