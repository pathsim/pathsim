"""Dynamic subsystems must see the parent network's current stage inputs."""

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Amplifier, Integrator, Scope
from pathsim.subsystem import Subsystem, Interface
from pathsim.solvers import EUF, SSPRK22, RKCK54, ESDIRK32


class TestSubsystemCurrentInputs(unittest.TestCase):

    def diagram(self, depth, Solver):
        source, gain = Source(lambda t: 1. + t), Amplifier(2.)
        inner_gain, integ, scope = Amplifier(3.), Integrator(), Scope()
        interface = Interface()
        wrapper = Subsystem(
            [interface, inner_gain, integ],
            [Connection(interface, inner_gain), Connection(inner_gain, integ),
             Connection(integ, interface)]
            )
        for _ in range(depth - 1):
            interface = Interface()
            wrapper = Subsystem(
                [interface, wrapper],
                [Connection(interface, wrapper), Connection(wrapper, interface)]
                )
        sim = Simulation(
            [source, gain, wrapper, scope],
            [Connection(source, gain), Connection(gain, wrapper), Connection(wrapper, scope)],
            Solver=Solver, dt=.01, log=False
            )
        return sim, wrapper, integ

    def test_derivative_uses_current_inputs(self):
        for depth in [1, 2, 3]:
            with self.subTest(depth=depth):
                sim, wrapper, _ = self.diagram(depth, SSPRK22)
                #dx/dt=6*(1+t), independent of any previous evaluation
                for t in [.5, 1., 0.]:
                    sim._update(t)
                    np.testing.assert_allclose(wrapper.derivative(t), [6.*(1.+t)], atol=1e-12)

    def test_explicit_stage_inputs(self):
        for Solver in [SSPRK22, RKCK54]:
            for depth in [1, 2, 3]:
                with self.subTest(Solver=Solver.__name__, depth=depth):
                    sim, _, integ = self.diagram(depth, Solver)
                    sim.run(1., adaptive=False)
                    #integral from 0 to 1 of 6*(1+t) is exactly 9
                    np.testing.assert_allclose(integ.state, [9.], atol=1e-11)

    def test_euler_and_implicit_controls(self):
        for Solver, expected in [(EUF, 8.97), (ESDIRK32, 9.)]:
            with self.subTest(Solver=Solver.__name__):
                sim, _, integ = self.diagram(2, Solver)
                sim.run(1., adaptive=False)
                np.testing.assert_allclose(integ.state, [expected], atol=1e-11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
