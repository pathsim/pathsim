########################################################################################
##
##                    Testing DynamicalSystem block in simulation
##
##   Verifies the general nonlinear state-space block (DynamicalSystem) produces
##   correct results for known analytical solutions when embedded in a simulation.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Source, Adder, Amplifier, Scope, DynamicalSystem, ODE

from pathsim.solvers import (
    RKBS32, RKCK54, RKDP54, RKV65, RKDP87,
    ESDIRK32, ESDIRK43
    )


# TESTCASE =============================================================================

class TestDynamicalSystemDecay(unittest.TestCase):
    """
    Test DynamicalSystem block with first-order linear decay.

    System: dx/dt = -a*x, y = x, x(0) = x0
    Analytical: x(t) = x0 * exp(-a*t)
    """

    def setUp(self):

        self.a = 2.0
        self.x0 = 3.0

        self.DS = DynamicalSystem(
            func_dyn=lambda x, u, t: -self.a * x,
            func_alg=lambda x, u, t: x,
            initial_value=self.x0
            )

        self.Sco = Scope(labels=["state"])

        self.Sim = Simulation(
            blocks=[self.DS, self.Sco],
            connections=[Connection(self.DS, self.Sco)],
            log=False
            )


    def _reference(self, t):
        return self.x0 * np.exp(-self.a * t)


    def test_eval_explicit_solvers(self):

        for SOL in [RKBS32, RKCK54, RKV65, RKDP87]:

            for tol in [1e-4, 1e-6, 1e-8]:

                with self.subTest(SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(SOL, tolerance_lte_abs=tol, tolerance_lte_rel=0.0)
                    self.Sim.run(5)

                    time, [res] = self.Sco.read()
                    ref = self._reference(time)

                    self.assertAlmostEqual(np.max(abs(ref - res)), tol, 2)


    def test_eval_implicit_solvers(self):

        for SOL in [ESDIRK32, ESDIRK43]:

            for tol in [1e-4, 1e-6]:

                with self.subTest(SOL=str(SOL), tol=tol):

                    self.Sim.reset()
                    self.Sim._set_solver(SOL, tolerance_lte_abs=tol, tolerance_lte_rel=0.0)
                    self.Sim.run(5)

                    time, [res] = self.Sco.read()
                    ref = self._reference(time)

                    self.assertAlmostEqual(np.max(abs(ref - res)), tol, 2)


class TestDynamicalSystemDriven(unittest.TestCase):
    """
    Test DynamicalSystem with external forcing input.

    System: dx/dt = -x + u, y = 2*x, u = 1 (step input)
    Analytical: x(t) = 1 - exp(-t), y(t) = 2*(1 - exp(-t))
    """

    def setUp(self):

        Src = Source(lambda t: 1.0)

        self.DS = DynamicalSystem(
            func_dyn=lambda x, u, t: -x + u,
            func_alg=lambda x, u, t: 2 * x,
            initial_value=0.0
            )

        self.Sco = Scope(labels=["output"])

        self.Sim = Simulation(
            blocks=[Src, self.DS, self.Sco],
            connections=[
                Connection(Src, self.DS),
                Connection(self.DS, self.Sco)
                ],
            log=False
            )


    def test_step_response(self):
        """Verify step response matches analytical solution"""

        for SOL in [RKCK54, RKDP87]:

            with self.subTest(SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6, tolerance_lte_rel=0.0)
                self.Sim.run(8)

                time, [res] = self.Sco.read()
                ref = 2.0 * (1.0 - np.exp(-time))

                error = np.max(np.abs(ref - res))
                self.assertLess(error, 1e-4,
                    f"Step response error: {error:.2e}")


class TestODEBlockInSimulation(unittest.TestCase):
    """
    Test the ODE block in a feedback simulation.

    System: dx/dt = -x + u, y = x, with u from a source
    This is similar to an integrator with feedback, but using the
    general ODE block.
    """

    def setUp(self):

        Src = Source(lambda t: np.sin(t))

        self.Ode = ODE(
            func=lambda x, u, t: -x + u,
            initial_value=0.0
            )

        self.Sco = Scope(labels=["ode_output", "source"])

        self.Sim = Simulation(
            blocks=[Src, self.Ode, self.Sco],
            connections=[
                Connection(Src, self.Ode, self.Sco[1]),
                Connection(self.Ode, self.Sco[0])
                ],
            log=False
            )


    def test_sinusoidal_response(self):
        """Test ODE response to sinusoidal input"""

        self.Sim._set_solver(RKCK54, tolerance_lte_abs=1e-8, tolerance_lte_rel=0.0)
        self.Sim.run(duration=20, reset=True)

        time, [ode_out, src_out] = self.Sco.read()

        #analytical solution for dx/dt = -x + sin(t), x(0) = 0:
        # x(t) = 0.5*(sin(t) - cos(t)) + 0.5*exp(-t)
        ref = 0.5 * (np.sin(time) - np.cos(time)) + 0.5 * np.exp(-time)

        error = np.max(np.abs(ref - ode_out))
        self.assertLess(error, 1e-5,
            f"ODE sinusoidal response error: {error:.2e}")


class TestDynamicalSystemFeedback(unittest.TestCase):
    """
    Test DynamicalSystem in a feedback loop with an integrator.

    This tests a more complex topology where DynamicalSystem interacts
    with other blocks through connections.
    """

    def test_coupled_system(self):
        """Two coupled first-order systems"""

        #system 1: dx1/dt = -x1 + u1, y1 = x1
        DS1 = DynamicalSystem(
            func_dyn=lambda x, u, t: -x + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0
            )

        #system 2: dx2/dt = -2*x2 + u2, y2 = x2
        DS2 = DynamicalSystem(
            func_dyn=lambda x, u, t: -2*x + u,
            func_alg=lambda x, u, t: x,
            initial_value=0.0
            )

        Sco = Scope(labels=["x1", "x2"])

        #DS1 output feeds DS2, DS2 output feeds DS1
        Sim = Simulation(
            blocks=[DS1, DS2, Sco],
            connections=[
                Connection(DS1, DS2, Sco[0]),
                Connection(DS2, DS1, Sco[1])
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-8,
            log=False
            )

        Sim.run(duration=10, reset=True)

        time, [x1, x2] = Sco.read()

        #both states should decay toward 0 (coupled decay)
        self.assertAlmostEqual(x1[-1], 0.0, 1)
        self.assertAlmostEqual(x2[-1], 0.0, 1)

        #x1 should start at 1.0
        self.assertAlmostEqual(x1[0], 1.0, 4)

        #x2 should start at 0.0
        self.assertAlmostEqual(x2[0], 0.0, 4)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
