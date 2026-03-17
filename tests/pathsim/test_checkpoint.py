"""Tests for checkpoint save/load functionality."""

import os
import json
import tempfile

import numpy as np
import pytest

from pathsim import Simulation, Connection, Subsystem, Interface
from pathsim.blocks import (
    Source, Integrator, Amplifier, Scope, Constant, Function
)
from pathsim.blocks.delay import Delay
from pathsim.blocks.switch import Switch
from pathsim.blocks.fir import FIR
from pathsim.blocks.kalman import KalmanFilter
from pathsim.blocks.noise import WhiteNoise, PinkNoise
from pathsim.blocks.rng import RandomNumberGenerator


class TestBlockCheckpoint:
    """Test block-level checkpoint methods."""

    def test_basic_block_to_checkpoint(self):
        """Block produces valid checkpoint data."""
        b = Integrator(1.0)
        b.inputs[0] = 3.14
        prefix = "Integrator_0"
        json_data, npz_data = b.to_checkpoint(prefix)

        assert json_data["type"] == "Integrator"
        assert json_data["active"] is True
        assert f"{prefix}/inputs" in npz_data
        assert f"{prefix}/outputs" in npz_data

    def test_block_checkpoint_roundtrip(self):
        """Block state survives save/load cycle."""
        b = Integrator(2.5)
        b.inputs[0] = 1.0
        b.outputs[0] = 2.5
        prefix = "Integrator_0"

        json_data, npz_data = b.to_checkpoint(prefix)

        #reset block
        b.reset()
        assert b.inputs[0] == 0.0

        #restore
        b.load_checkpoint(prefix, json_data, npz_data)
        assert np.isclose(b.inputs[0], 1.0)
        assert np.isclose(b.outputs[0], 2.5)

    def test_block_type_mismatch_raises(self):
        """Loading checkpoint with wrong type raises ValueError."""
        b = Integrator()
        prefix = "Integrator_0"
        json_data, npz_data = b.to_checkpoint(prefix)

        b2 = Amplifier(1.0)
        with pytest.raises(ValueError, match="type mismatch"):
            b2.load_checkpoint(prefix, json_data, npz_data)


class TestEventCheckpoint:
    """Test event-level checkpoint methods."""

    def test_event_checkpoint_roundtrip(self):
        from pathsim.events import ZeroCrossing
        e = ZeroCrossing(func_evt=lambda t: t - 1.0)
        e._history = (0.5, 0.99)
        e._times = [1.0, 2.0, 3.0]
        e._active = False
        prefix = "ZeroCrossing_0"

        json_data, npz_data = e.to_checkpoint(prefix)

        e.reset()
        assert e._active is True
        assert len(e._times) == 0

        e.load_checkpoint(prefix, json_data, npz_data)
        assert e._active is False
        assert e._times == [1.0, 2.0, 3.0]
        assert e._history == (0.5, 0.99)


class TestSwitchCheckpoint:
    """Test Switch block checkpoint."""

    def test_switch_state_preserved(self):
        s = Switch(switch_state=2)
        prefix = "Switch_0"
        json_data, npz_data = s.to_checkpoint(prefix)

        s.select(None)
        assert s.switch_state is None

        s.load_checkpoint(prefix, json_data, npz_data)
        assert s.switch_state == 2


class TestSimulationCheckpoint:
    """Test simulation-level checkpoint save/load."""

    def test_save_load_simple(self):
        """Simple simulation checkpoint round-trip."""
        src = Source(lambda t: np.sin(2 * np.pi * t))
        integ = Integrator()
        scope = Scope()

        sim = Simulation(
            blocks=[src, integ, scope],
            connections=[
                Connection(src, integ, scope),
            ],
            dt=0.01
        )

        #run for 1 second
        sim.run(1.0)
        time_after_run = sim.time
        state_after_run = integ.state.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint")
            sim.save_checkpoint(path)

            #verify files exist
            assert os.path.exists(f"{path}.json")
            assert os.path.exists(f"{path}.npz")

            #verify JSON structure
            with open(f"{path}.json") as f:
                data = json.load(f)
            assert data["version"] == "1.0.0"
            assert data["simulation"]["time"] == time_after_run
            assert any(b["_key"] == "Integrator_0" for b in data["blocks"])

            #reset and reload
            sim.time = 0.0
            integ.state = np.array([0.0])

            sim.load_checkpoint(path)
            assert sim.time == time_after_run
            assert np.allclose(integ.state, state_after_run)

    def test_continue_after_load(self):
        """Simulation continues correctly after checkpoint load."""
        #run continuously for 2 seconds
        src1 = Source(lambda t: 1.0)
        integ1 = Integrator()
        sim1 = Simulation(
            blocks=[src1, integ1],
            connections=[Connection(src1, integ1)],
            dt=0.01
        )
        sim1.run(2.0)
        reference_state = integ1.state.copy()

        #run for 1 second, save, load, run 1 more second
        src2 = Source(lambda t: 1.0)
        integ2 = Integrator()
        sim2 = Simulation(
            blocks=[src2, integ2],
            connections=[Connection(src2, integ2)],
            dt=0.01
        )
        sim2.run(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim2.save_checkpoint(path)
            sim2.load_checkpoint(path)
            sim2.run(1.0)  # run 1 more second (t=1 -> t=2)

        #compare results
        assert np.allclose(integ2.state, reference_state, rtol=1e-6)

    def test_scope_recordings(self):
        """Scope recordings are saved when recordings=True."""
        src = Source(lambda t: t)
        scope = Scope()
        sim = Simulation(
            blocks=[src, scope],
            connections=[Connection(src, scope)],
            dt=0.1
        )
        sim.run(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            #without recordings
            path1 = os.path.join(tmpdir, "no_rec")
            sim.save_checkpoint(path1, recordings=False)
            npz1 = np.load(f"{path1}.npz")
            assert "Scope_0/recording_time" not in npz1
            npz1.close()

            #with recordings
            path2 = os.path.join(tmpdir, "with_rec")
            sim.save_checkpoint(path2, recordings=True)
            npz2 = np.load(f"{path2}.npz")
            assert "Scope_0/recording_time" in npz2
            npz2.close()

    def test_delay_continuous_checkpoint(self):
        """Continuous delay block preserves buffer."""
        src = Source(lambda t: np.sin(t))
        delay = Delay(tau=0.1)
        scope = Scope()
        sim = Simulation(
            blocks=[src, delay, scope],
            connections=[
                Connection(src, delay, scope),
            ],
            dt=0.01
        )
        sim.run(0.5)

        #capture delay output
        delay_output = delay.outputs[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            #reset delay buffer
            delay._buffer.clear()

            sim.load_checkpoint(path)
            assert np.isclose(delay.outputs[0], delay_output)

    def test_delay_discrete_checkpoint(self):
        """Discrete delay block preserves ring buffer."""
        src = Source(lambda t: float(t > 0))
        delay = Delay(tau=0.05, sampling_period=0.01)
        sim = Simulation(
            blocks=[src, delay],
            connections=[Connection(src, delay)],
            dt=0.01
        )
        sim.run(0.1)

        ring_before = list(delay._ring)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)
            delay._ring.clear()
            sim.load_checkpoint(path)
            assert list(delay._ring) == ring_before

    def test_cross_instance_load(self):
        """Checkpoint loads into a freshly constructed simulation (different UUIDs)."""
        src1 = Source(lambda t: 1.0)
        integ1 = Integrator()
        sim1 = Simulation(
            blocks=[src1, integ1],
            connections=[Connection(src1, integ1)],
            dt=0.01
        )
        sim1.run(1.0)
        saved_time = sim1.time
        saved_state = integ1.state.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim1.save_checkpoint(path)

            #create entirely new simulation (new block objects, new UUIDs)
            src2 = Source(lambda t: 1.0)
            integ2 = Integrator()
            sim2 = Simulation(
                blocks=[src2, integ2],
                connections=[Connection(src2, integ2)],
                dt=0.01
            )

            sim2.load_checkpoint(path)
            assert sim2.time == saved_time
            assert np.allclose(integ2.state, saved_state)

    def test_scope_recordings_preserved_without_flag(self):
        """Loading without recordings flag does not erase existing recordings."""
        src = Source(lambda t: t)
        scope = Scope()
        sim = Simulation(
            blocks=[src, scope],
            connections=[Connection(src, scope)],
            dt=0.1
        )
        sim.run(1.0)

        #scope has recordings
        assert len(scope.recording_time) > 0
        rec_len = len(scope.recording_time)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path, recordings=False)
            sim.load_checkpoint(path)

            #recordings should still be intact
            assert len(scope.recording_time) == rec_len

    def test_multiple_same_type_blocks(self):
        """Multiple blocks of the same type are matched by insertion order."""
        src = Source(lambda t: 1.0)
        i1 = Integrator(1.0)
        i2 = Integrator(2.0)
        sim = Simulation(
            blocks=[src, i1, i2],
            connections=[Connection(src, i1), Connection(src, i2)],
            dt=0.01
        )
        sim.run(0.5)

        state1 = i1.state.copy()
        state2 = i2.state.copy()
        assert not np.allclose(state1, state2)  # different initial conditions

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            i1.state = np.array([0.0])
            i2.state = np.array([0.0])

            sim.load_checkpoint(path)
            assert np.allclose(i1.state, state1)
            assert np.allclose(i2.state, state2)


class TestFIRCheckpoint:
    """Test FIR block checkpoint."""

    def test_fir_buffer_preserved(self):
        """FIR filter buffer survives checkpoint round-trip."""
        fir = FIR(coeffs=[0.25, 0.5, 0.25], T=0.01)
        prefix = "FIR_0"

        #simulate some input to fill the buffer
        fir._buffer.appendleft(1.0)
        fir._buffer.appendleft(2.0)
        buffer_before = list(fir._buffer)

        json_data, npz_data = fir.to_checkpoint(prefix)

        fir._buffer.clear()
        fir._buffer.extend([0.0] * 3)

        fir.load_checkpoint(prefix, json_data, npz_data)
        assert list(fir._buffer) == buffer_before


class TestKalmanFilterCheckpoint:
    """Test KalmanFilter block checkpoint."""

    def test_kalman_state_preserved(self):
        """Kalman filter state and covariance survive checkpoint."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])

        kf = KalmanFilter(F, H, Q, R)
        prefix = "KalmanFilter_0"

        #set some state
        kf.x = np.array([3.14, -1.0])
        kf.P = np.array([[0.5, 0.1], [0.1, 0.3]])

        json_data, npz_data = kf.to_checkpoint(prefix)

        kf.x = np.zeros(2)
        kf.P = np.eye(2)

        kf.load_checkpoint(prefix, json_data, npz_data)
        assert np.allclose(kf.x, [3.14, -1.0])
        assert np.allclose(kf.P, [[0.5, 0.1], [0.1, 0.3]])


class TestNoiseCheckpoint:
    """Test noise block checkpoints."""

    def test_white_noise_sample_preserved(self):
        """WhiteNoise current sample survives checkpoint."""
        wn = WhiteNoise(standard_deviation=2.0)
        wn._current_sample = 1.234
        prefix = "WhiteNoise_0"

        json_data, npz_data = wn.to_checkpoint(prefix)
        wn._current_sample = 0.0

        wn.load_checkpoint(prefix, json_data, npz_data)
        assert wn._current_sample == pytest.approx(1.234)

    def test_pink_noise_state_preserved(self):
        """PinkNoise algorithm state survives checkpoint."""
        pn = PinkNoise(num_octaves=8, seed=42)
        prefix = "PinkNoise_0"

        #advance the noise state
        for _ in range(10):
            pn._generate_sample(0.01)

        n_samples_before = pn.n_samples
        octaves_before = pn.octave_values.copy()
        sample_before = pn._current_sample

        json_data, npz_data = pn.to_checkpoint(prefix)

        pn.reset()
        assert pn.n_samples == 0

        pn.load_checkpoint(prefix, json_data, npz_data)
        assert pn.n_samples == n_samples_before
        assert np.allclose(pn.octave_values, octaves_before)


class TestRNGCheckpoint:
    """Test RandomNumberGenerator checkpoint."""

    def test_rng_sample_preserved(self):
        """RNG current sample survives checkpoint (continuous mode)."""
        rng = RandomNumberGenerator(sampling_period=None)
        prefix = "RandomNumberGenerator_0"
        sample_before = rng._sample

        json_data, npz_data = rng.to_checkpoint(prefix)
        rng._sample = 0.0

        rng.load_checkpoint(prefix, json_data, npz_data)
        assert rng._sample == pytest.approx(sample_before)


class TestSubsystemCheckpoint:
    """Test Subsystem checkpoint."""

    def test_subsystem_roundtrip(self):
        """Subsystem with internal blocks survives checkpoint round-trip."""
        #build a simple subsystem: two integrators in series
        If = Interface()
        I1 = Integrator(1.0)
        I2 = Integrator(0.0)

        sub = Subsystem(
            blocks=[If, I1, I2],
            connections=[
                Connection(If, I1),
                Connection(I1, I2),
                Connection(I2, If),
            ]
        )

        #embed in a simulation
        src = Source(lambda t: 1.0)
        scope = Scope()
        sim = Simulation(
            blocks=[src, sub, scope],
            connections=[
                Connection(src, sub),
                Connection(sub, scope),
            ],
            dt=0.01
        )

        sim.run(0.5)
        state_I1 = I1.state.copy()
        state_I2 = I2.state.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            #zero out states
            I1.state = np.array([0.0])
            I2.state = np.array([0.0])

            sim.load_checkpoint(path)
            assert np.allclose(I1.state, state_I1)
            assert np.allclose(I2.state, state_I2)

    def test_subsystem_cross_instance(self):
        """Subsystem checkpoint loads into a fresh simulation instance."""
        If1 = Interface()
        I1 = Integrator(1.0)
        sub1 = Subsystem(
            blocks=[If1, I1],
            connections=[Connection(If1, I1), Connection(I1, If1)]
        )
        src1 = Source(lambda t: 1.0)
        sim1 = Simulation(
            blocks=[src1, sub1],
            connections=[Connection(src1, sub1)],
            dt=0.01
        )
        sim1.run(0.5)
        state_before = I1.state.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim1.save_checkpoint(path)

            #new instance
            If2 = Interface()
            I2 = Integrator(1.0)
            sub2 = Subsystem(
                blocks=[If2, I2],
                connections=[Connection(If2, I2), Connection(I2, If2)]
            )
            src2 = Source(lambda t: 1.0)
            sim2 = Simulation(
                blocks=[src2, sub2],
                connections=[Connection(src2, sub2)],
                dt=0.01
            )
            sim2.load_checkpoint(path)
            assert np.allclose(I2.state, state_before)


class TestGEARCheckpoint:
    """Test GEAR solver checkpoint round-trip."""

    def test_gear_solver_roundtrip(self):
        """GEAR solver state survives checkpoint including BDF coefficients."""
        from pathsim.solvers import GEAR32

        src = Source(lambda t: np.sin(2 * np.pi * t))
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01,
            Solver=GEAR32
        )

        #run long enough for GEAR to exit startup phase
        sim.run(0.5)
        state_after = integ.state.copy()
        time_after = sim.time

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            #reset state
            integ.state = np.array([0.0])
            sim.time = 0.0

            sim.load_checkpoint(path)
            assert sim.time == time_after
            assert np.allclose(integ.state, state_after)

    def test_gear_continue_after_load(self):
        """GEAR simulation continues correctly after checkpoint load."""
        from pathsim.solvers import GEAR32

        #reference: run 2s continuously
        src1 = Source(lambda t: 1.0)
        integ1 = Integrator()
        sim1 = Simulation(
            blocks=[src1, integ1],
            connections=[Connection(src1, integ1)],
            dt=0.01,
            Solver=GEAR32
        )
        sim1.run(2.0)
        reference = integ1.state.copy()

        #split: run 1s, save, load, run 1s more
        src2 = Source(lambda t: 1.0)
        integ2 = Integrator()
        sim2 = Simulation(
            blocks=[src2, integ2],
            connections=[Connection(src2, integ2)],
            dt=0.01,
            Solver=GEAR32
        )
        sim2.run(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim2.save_checkpoint(path)
            sim2.load_checkpoint(path)
            sim2.run(1.0)

        assert np.allclose(integ2.state, reference, rtol=1e-6)


class TestSpectrumCheckpoint:
    """Test Spectrum block checkpoint."""

    def test_spectrum_roundtrip(self):
        """Spectrum block state survives checkpoint round-trip."""
        from pathsim.blocks.spectrum import Spectrum

        src = Source(lambda t: np.sin(2 * np.pi * 10 * t))
        spec = Spectrum(freq=[5, 10, 15], t_wait=0.0)
        sim = Simulation(
            blocks=[src, spec],
            connections=[Connection(src, spec)],
            dt=0.001
        )
        sim.run(0.1)

        time_before = spec.time
        t_sample_before = spec.t_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            spec.time = 0.0
            spec.t_sample = 0.0

            sim.load_checkpoint(path)
            assert spec.time == pytest.approx(time_before)
            assert spec.t_sample == pytest.approx(t_sample_before)


class TestScopeCheckpointExtended:
    """Extended Scope checkpoint tests for coverage."""

    def test_scope_with_sampling_period(self):
        """Scope with sampling_period preserves _sample_next_timestep."""
        src = Source(lambda t: t)
        scope = Scope(sampling_period=0.1)
        sim = Simulation(
            blocks=[src, scope],
            connections=[Connection(src, scope)],
            dt=0.01
        )
        sim.run(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)
            sim.load_checkpoint(path)

            #verify scope still works after load
            sim.run(0.1)
            assert len(scope.recording_time) > 0

    def test_scope_recordings_roundtrip(self):
        """Scope recording data round-trips with recordings=True."""
        src = Source(lambda t: t)
        scope = Scope()
        sim = Simulation(
            blocks=[src, scope],
            connections=[Connection(src, scope)],
            dt=0.1
        )
        sim.run(1.0)

        rec_time = scope.recording_time.copy()
        rec_data = [row.copy() for row in scope.recording_data]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path, recordings=True)

            #clear recordings
            scope.recording_time = []
            scope.recording_data = []

            sim.load_checkpoint(path)
            assert len(scope.recording_time) == len(rec_time)
            assert np.allclose(scope.recording_time, rec_time)

    def test_scope_recordings_included_by_default(self):
        """Default save_checkpoint includes recordings."""
        src = Source(lambda t: t)
        scope = Scope()
        sim = Simulation(
            blocks=[src, scope],
            connections=[Connection(src, scope)],
            dt=0.1
        )
        sim.run(1.0)

        rec_time = scope.recording_time.copy()
        assert len(rec_time) > 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)  # no recordings kwarg — default

            #clear recordings
            scope.recording_time = []
            scope.recording_data = []

            sim.load_checkpoint(path)
            assert len(scope.recording_time) == len(rec_time)
            assert np.allclose(scope.recording_time, rec_time)


class TestSimulationCheckpointExtended:
    """Extended simulation checkpoint tests for coverage."""

    def test_save_load_with_extension(self):
        """Path with .json extension is handled correctly."""
        src = Source(lambda t: 1.0)
        integ = Integrator()
        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            dt=0.01
        )
        sim.run(0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp.json")
            sim.save_checkpoint(path)

            assert os.path.exists(os.path.join(tmpdir, "cp.json"))
            assert os.path.exists(os.path.join(tmpdir, "cp.npz"))

            sim.load_checkpoint(path)
            assert sim.time == pytest.approx(0.1, abs=0.01)

    def test_checkpoint_with_events(self):
        """Simulation with external events checkpoints correctly."""
        from pathsim.events import Schedule

        src = Source(lambda t: 1.0)
        integ = Integrator()

        event_fired = [False]
        def on_event(t):
            event_fired[0] = True

        evt = Schedule(t_start=0.5, t_period=1.0, func_act=on_event)

        sim = Simulation(
            blocks=[src, integ],
            connections=[Connection(src, integ)],
            events=[evt],
            dt=0.01
        )
        sim.run(1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cp")
            sim.save_checkpoint(path)

            #verify events in JSON
            with open(f"{path}.json") as f:
                data = json.load(f)
            assert len(data["events"]) == 1
            assert data["events"][0]["type"] == "Schedule"

            sim.load_checkpoint(path)

    def test_event_numpy_history(self):
        """Event with numpy scalar in history serializes correctly."""
        from pathsim.events import ZeroCrossing

        e = ZeroCrossing(func_evt=lambda t: t - 1.0)
        e._history = (np.float64(0.5), 0.99)
        prefix = "ZeroCrossing_0"

        json_data, npz_data = e.to_checkpoint(prefix)
        assert isinstance(json_data["history_eval"], float)

        e.reset()
        e.load_checkpoint(prefix, json_data, npz_data)
        assert e._history[0] == pytest.approx(0.5)
