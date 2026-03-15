"""Tests for checkpoint save/load functionality."""

import os
import json
import tempfile

import numpy as np
import pytest

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source, Integrator, Amplifier, Scope, Constant
)
from pathsim.blocks.delay import Delay
from pathsim.blocks.switch import Switch


class TestBlockCheckpoint:
    """Test block-level checkpoint methods."""

    def test_basic_block_to_checkpoint(self):
        """Block produces valid checkpoint data."""
        b = Integrator(1.0)
        b.inputs[0] = 3.14
        json_data, npz_data = b.to_checkpoint()

        assert json_data["type"] == "Integrator"
        assert json_data["id"] == b.id
        assert json_data["active"] is True
        assert f"{b.id}/inputs" in npz_data
        assert f"{b.id}/outputs" in npz_data

    def test_block_has_uuid(self):
        """Each block gets a unique UUID."""
        b1 = Integrator()
        b2 = Integrator()
        assert b1.id != b2.id
        assert len(b1.id) == 32  # hex UUID without dashes

    def test_block_checkpoint_roundtrip(self):
        """Block state survives save/load cycle."""
        b = Integrator(2.5)
        b.inputs[0] = 1.0
        b.outputs[0] = 2.5

        json_data, npz_data = b.to_checkpoint()

        #reset block
        b.reset()
        assert b.inputs[0] == 0.0

        #restore
        b.load_checkpoint(json_data, npz_data)
        assert np.isclose(b.inputs[0], 1.0)
        assert np.isclose(b.outputs[0], 2.5)

    def test_block_type_mismatch_raises(self):
        """Loading checkpoint with wrong type raises ValueError."""
        b = Integrator()
        json_data, npz_data = b.to_checkpoint()

        b2 = Amplifier(1.0)
        with pytest.raises(ValueError, match="type mismatch"):
            b2.load_checkpoint(json_data, npz_data)


class TestEventCheckpoint:
    """Test event-level checkpoint methods."""

    def test_event_has_uuid(self):
        from pathsim.events import ZeroCrossing
        e = ZeroCrossing(func_evt=lambda t: t - 1.0)
        assert len(e.id) == 32

    def test_event_checkpoint_roundtrip(self):
        from pathsim.events import ZeroCrossing
        e = ZeroCrossing(func_evt=lambda t: t - 1.0)
        e._history = (0.5, 0.99)
        e._times = [1.0, 2.0, 3.0]
        e._active = False

        json_data, npz_data = e.to_checkpoint()

        e.reset()
        assert e._active is True
        assert len(e._times) == 0

        e.load_checkpoint(json_data, npz_data)
        assert e._active is False
        assert e._times == [1.0, 2.0, 3.0]
        assert e._history == (0.5, 0.99)


class TestSwitchCheckpoint:
    """Test Switch block checkpoint."""

    def test_switch_state_preserved(self):
        s = Switch(switch_state=2)
        json_data, npz_data = s.to_checkpoint()

        s.select(None)
        assert s.switch_state is None

        s.load_checkpoint(json_data, npz_data)
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
            assert integ.id in data["blocks"]

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
            assert f"{scope.id}/recording_time" not in npz1
            npz1.close()

            #with recordings
            path2 = os.path.join(tmpdir, "with_rec")
            sim.save_checkpoint(path2, recordings=True)
            npz2 = np.load(f"{path2}.npz")
            assert f"{scope.id}/recording_time" in npz2
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
