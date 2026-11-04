"""Multi-channel discrete delay, including a partially filled checkpoint."""

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Delay, Scope, Source
from pathsim.solvers import EUF, RKBS32


class TestDelayChannels(unittest.TestCase):

    def test_discrete_channels(self):

        values = np.array([[2., -3.], [0., 5.], [-1., 0.5], [4., -2.],
                           [0.5, 0.], [1., 2.], [-0.5, 3.], [0., -1.]])
        ts = 0.001
        for n in [1, 3]:
            for solver in [EUF, RKBS32]:
                with self.subTest(n=n, solver=solver):
                    sources = [Source(lambda t, c=c: values[min(
                        int(np.floor(t / ts + 1e-9)), len(values)-1), c])
                        for c in range(2)]
                    delay = Delay(tau=n*ts, sampling_period=ts)
                    scope = Scope(sampling_period=ts, t_wait=ts/2)
                    connections = [Connection(sources[c], delay[c]) for c in range(2)]
                    connections.extend(Connection(delay[c], scope[c]) for c in range(2))
                    sim = Simulation(sources + [delay, scope], connections,
                                     dt=ts/4, dt_max=ts/4, Solver=solver, log=False)
                    sim.run(len(values)*ts)
                    _, data = scope.read()
                    expected = np.concatenate((np.zeros((n, 2)), values[:-n])).T
                    np.testing.assert_array_equal(data, expected)


    def test_checkpoint_before_ring_fills(self):

        delay = Delay(tau=3, sampling_period=1)
        delay.inputs.update_from_array([2., -3.])
        delay._sample_next_timestep = True
        delay.sample(0, 1)

        #later input changes must not alter a captured channel vector
        delay.inputs[1] = 99.
        metadata, arrays = delay.to_checkpoint("D")
        restored = Delay(tau=3, sampling_period=1)
        restored.load_checkpoint("D", metadata, arrays)

        for k in range(1, 4):
            restored.inputs.update_from_array([0., 0.])
            restored._sample_next_timestep = True
            restored.sample(k, 1)
        restored.update(3)
        np.testing.assert_array_equal(restored.outputs.to_array(), [2., -3.])


    def test_scalar_checkpoint_shape(self):

        delay = Delay(tau=3, sampling_period=1)
        delay.inputs[0] = 2.
        delay._sample_next_timestep = True
        delay.sample(0, 1)
        _, arrays = delay.to_checkpoint("D")
        np.testing.assert_array_equal(arrays["D/ring"], [0., 0., 0., 2.])
