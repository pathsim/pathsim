########################################################################################
##
##                                  TESTS FOR
##                       'Simulation.pss' (periodic steady state)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import ODE, Constant, Amplifier, Scope
from pathsim.solvers.esdirk43 import ESDIRK43


# TESTS ================================================================================

class TestPeriodicSteadyState(unittest.TestCase):
    """
    Test the 'Simulation.pss' periodic steady state shooting method.
    """

    def test_forced_linear_scalar(self):
        #x' = -x + sin(t), periodic solution x_p = (sin t - cos t)/2, x_p(0) = -0.5
        ode = ODE(lambda x, u, t: -x + np.sin(t), 0.0)
        sim = Simulation(blocks=[ode], dt=0.01, Solver=ESDIRK43, log=False)

        success, iters, res = sim.pss(period=2*np.pi, dt=0.01)

        self.assertTrue(success)
        self.assertLess(res, 1e-8)
        self.assertAlmostEqual(float(ode.engine.state[0]), -0.5, places=6)


    def test_time_reset_to_cycle_start(self):
        ode = ODE(lambda x, u, t: -x + np.sin(t), 0.0)
        sim = Simulation(blocks=[ode], dt=0.01, Solver=ESDIRK43, log=False)
        sim.pss(period=2*np.pi, dt=0.01)
        #the simulation time is left at the cycle start
        self.assertAlmostEqual(sim.time, 0.0, places=12)


    def test_self_consistent_vector(self):
        #damped driven oscillator, x = [pos, vel]
        def func(x, u, t):
            return np.array([x[1], -x[0] - 0.3*x[1] + np.cos(t)])

        ode = ODE(func, [0.0, 0.0])
        sim = Simulation(blocks=[ode], dt=0.005, Solver=ESDIRK43, log=False)

        success, iters, res = sim.pss(period=2*np.pi, dt=0.005)
        self.assertTrue(success)

        #the found state must reproduce itself after exactly one period
        x0 = ode.engine.state.copy()
        sim._pss_integrate(sim.time, 2*np.pi, 0.005)
        np.testing.assert_allclose(ode.engine.state, x0, atol=1e-4)


    def test_sampling_suppressed_during_shooting(self):
        #scopes must not record during the shooting iterations
        ode = ODE(lambda x, u, t: -x + np.sin(t), 0.0)
        sco = Scope()
        sim = Simulation(
            blocks=[ode, sco],
            connections=[Connection(ode[0], sco[0])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.pss(period=2*np.pi, dt=0.01)
        self.assertEqual(len(sco.recording_time), 0)
        #sampling is re-enabled afterwards
        self.assertTrue(sim._sampling_enabled)


    def test_no_dynamic_states(self):
        #a purely algebraic system has no periodic steady state to solve
        src = Constant(1.0)
        amp = Amplifier(2.0)
        sim = Simulation(
            blocks=[src, amp],
            connections=[Connection(src, amp)],
            log=False
            )
        success, iters, res = sim.pss(period=1.0)
        self.assertTrue(success)
        self.assertEqual(iters, 0)


    def test_transient_then_shoot(self):
        #a transient run before shooting must still converge to the same cycle
        ode = ODE(lambda x, u, t: -x + np.sin(t), 5.0)
        sim = Simulation(blocks=[ode], dt=0.01, Solver=ESDIRK43, log=False)
        success, iters, res = sim.pss(period=2*np.pi, transient=10*np.pi, dt=0.01)
        self.assertTrue(success)
        #periodic solution x_p(t) = (sin t - cos t)/2 at the actual cycle start
        t0 = sim.time
        expected = 0.5 * (np.sin(t0) - np.cos(t0))
        self.assertAlmostEqual(float(ode.engine.state[0]), expected, places=5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
