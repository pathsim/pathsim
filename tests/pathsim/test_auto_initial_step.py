########################################################################################
##
##                                  TESTS FOR
##                    'Simulation.estimate_initial_step' / auto dt
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Integrator, Amplifier, Source, Scope


# TESTS ================================================================================

class TestBlockDerivative(unittest.TestCase):
    """
    Test the 'Block.derivative' accessor used for initial step estimation.
    """

    def test_ode_derivative(self):
        ode = ODE(lambda x, u, t: -2.0*x, 3.0)
        ode.set_solver(type(Simulation(log=False).engine), None)
        d = ode.derivative(0.0)
        np.testing.assert_allclose(np.atleast_1d(d), [-6.0])

    def test_integrator_derivative(self):
        itg = Integrator(0.0)
        itg.set_solver(type(Simulation(log=False).engine), None)
        itg.inputs[0] = 1.5
        np.testing.assert_allclose(itg.derivative(0.0), [1.5])

    def test_stateless_block_derivative_none(self):
        amp = Amplifier(2.0)
        self.assertIsNone(amp.derivative(0.0))


class TestEstimateInitialStep(unittest.TestCase):
    """
    Test the 'Simulation.estimate_initial_step' Hairer-Wanner estimator.
    """

    def test_scales_with_dynamics(self):
        #a faster system must get a smaller initial step
        sim_slow = Simulation(blocks=[ODE(lambda x, u, t: -x, 1.0)], log=False)
        sim_fast = Simulation(blocks=[ODE(lambda x, u, t: -100.0*x, 1.0)], log=False)
        h_slow = sim_slow.estimate_initial_step()
        h_fast = sim_fast.estimate_initial_step()
        self.assertGreater(h_slow, h_fast)
        #order of magnitude sanity: ~1/100 of the slow step
        self.assertLess(h_fast, h_slow / 10.0)

    def test_returns_positive(self):
        sim = Simulation(blocks=[ODE(lambda x, u, t: -x, 1.0)], log=False)
        self.assertGreater(sim.estimate_initial_step(), 0.0)

    def test_no_dynamic_states_returns_fallback(self):
        #purely algebraic system -> configured timestep is returned unchanged
        src = Source(lambda t: t)
        amp = Amplifier(2.0)
        sim = Simulation(
            blocks=[src, amp],
            connections=[Connection(src, amp)],
            dt=0.05, log=False
            )
        self.assertEqual(sim.estimate_initial_step(), 0.05)

    def test_state_restored(self):
        #the estimate must not mutate the block state
        ode = ODE(lambda x, u, t: -x, [1.0, 2.0, 3.0])
        sim = Simulation(blocks=[ode], log=False)
        sim.estimate_initial_step()
        np.testing.assert_array_equal(ode.engine.state, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(sim.time, 0.0)


class TestAutoTimestep(unittest.TestCase):
    """
    Test the automatic initial timestep selection via 'dt=None'.
    """

    def test_auto_dt_resolves_and_runs(self):
        #integrate cos(t) -> sin(t); dt=None must auto-select and run accurately
        src = Source(lambda t: np.cos(t))
        itg = Integrator(0.0)
        sco = Scope()
        sim = Simulation(
            blocks=[src, itg, sco],
            connections=[Connection(src, itg), Connection(itg, sco)],
            dt=None, log=False
            )
        self.assertIsNone(sim.dt)
        sim.run(1.0)
        #dt was resolved to a concrete positive value
        self.assertIsNotNone(sim.dt)
        self.assertGreater(sim.dt, 0.0)
        #result is accurate
        self.assertAlmostEqual(float(itg.engine.state[0]), np.sin(1.0), places=3)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
