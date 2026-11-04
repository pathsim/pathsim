"""Transient continuation after steady-state failures."""

import unittest
import numpy as np

from pathsim import Simulation, Connection, Subsystem, Interface
from pathsim.blocks import Constant, Integrator, ODE, Scope
from pathsim.solvers import RKCK54, SSPRK22


class TestSteadyStateRecovery(unittest.TestCase):

    def test_failed_solve_then_transient(self):
        #dx/dt=1 has no equilibrium, but its transient is well defined
        for Solver in [SSPRK22, RKCK54]:
            for reset in [False, True]:
                with self.subTest(Solver=Solver.__name__, reset=reset):
                    source, state, scope = Constant(1.), Integrator(), Scope()
                    sim = Simulation(
                        [source, state, scope],
                        [Connection(source, state), Connection(state, scope)],
                        Solver=Solver, iterations_max=3, log=False
                        )
                    with self.assertRaisesRegex(RuntimeError, "STEADYSTATE -> FAILED"):
                        sim.steadystate()
                    last_iterate = np.atleast_1d(state.state).copy()
                    self.assertIs(sim.Solver, Solver)
                    self.assertIsInstance(state.engine, Solver)
                    self.assertEqual(len(scope.recording_time), 0)
                    sim.run(.1, reset=reset, adaptive=False)
                    initial = 0. if reset else last_iterate
                    np.testing.assert_allclose(state.state, initial+.1, atol=1e-12)

    def test_failed_nested_solve_restores_engines(self):
        state = ODE(lambda x, u, t: np.ones_like(x))
        sub = Subsystem([Interface(), state], [])
        sim = Simulation([sub], Solver=RKCK54, iterations_max=3, log=False)
        with self.assertRaisesRegex(RuntimeError, "STEADYSTATE -> FAILED"):
            sim.steadystate()
        self.assertIs(sim.Solver, RKCK54)
        self.assertIsInstance(sub.engine, RKCK54)
        self.assertIsInstance(state.engine, RKCK54)
        sim.run(.1, reset=True)
        np.testing.assert_allclose(state.state, [.1], atol=1e-12)

    def test_rhs_exception_then_transient(self):
        failing = [True]

        def rhs(x, u, t):
            if failing[0]:
                raise ValueError("invalid operating point")
            return np.ones_like(x)

        state = ODE(rhs)
        sim = Simulation([state], Solver=RKCK54, log=False)
        with self.assertRaisesRegex(ValueError, "invalid operating point"):
            sim.steadystate()
        self.assertIs(sim.Solver, RKCK54)
        self.assertIsInstance(state.engine, RKCK54)
        failing[0] = False
        sim.run(.1)
        np.testing.assert_allclose(state.state, [.1], atol=1e-12)

    def test_sampling_exception_then_transient(self):
        class FailingScope(Scope):
            failing = True

            def sample(self, t, dt):
                if self.failing:
                    raise ValueError("sampling failed")
                return super().sample(t, dt)

        state, scope = ODE(lambda x, u, t: 1-x), FailingScope()
        sim = Simulation([state, scope], [Connection(state, scope)],
                         Solver=RKCK54, log=False)
        with self.assertRaisesRegex(ValueError, "sampling failed"):
            sim.steadystate()
        self.assertIs(sim.Solver, RKCK54)
        self.assertIsInstance(state.engine, RKCK54)
        scope.failing = False
        sim.run(.1, reset=False)
        np.testing.assert_allclose(state.state, [1.], atol=1e-12)

    def test_successful_solve_then_transient(self):
        state = ODE(lambda x, u, t: 1-x)
        sim = Simulation([state], Solver=RKCK54, log=False)
        sim.steadystate()
        self.assertIs(sim.Solver, RKCK54)
        sim.run(.1, reset=False)
        np.testing.assert_allclose(state.state, [1.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
