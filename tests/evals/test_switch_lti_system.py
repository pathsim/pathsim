########################################################################################
##
##                  Testing switch routing and LTI systems
##
##   Tests Switch block with scheduled switching, and StateSpace/TransferFunction
##   blocks in feedback loops. Verifies correct signal routing and LTI behavior.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    Constant,
    Switch,
    Integrator,
    Amplifier,
    Adder,
    Scope,
    StateSpace,
    TransferFunctionNumDen
    )

from pathsim.events.schedule import Schedule

from pathsim.solvers import RKCK54, ESDIRK32


# TESTCASE =============================================================================

class TestSwitchRoutingSystem(unittest.TestCase):
    """
    Test Switch block selecting between multiple input sources.

    System: two sources feed a switch, scheduled events toggle the switch.
    Verify: output matches the selected source at each time segment.
    """

    def test_switch_between_sources(self):
        """Switch between constant sources using scheduled events"""

        Src0 = Constant(1.0)
        Src1 = Constant(5.0)
        Sw = Switch(switch_state=0)  # start with input 0
        Sco = Scope(labels=["output"])

        #schedule to switch at t=2 and t=4
        def switch_to_1(t):
            Sw.select(1)
        def switch_to_0(t):
            Sw.select(0)

        evt1 = Schedule(t_start=2.0, t_period=100, func_act=switch_to_1)
        evt2 = Schedule(t_start=4.0, t_period=100, func_act=switch_to_0)

        Sim = Simulation(
            blocks=[Src0, Src1, Sw, Sco],
            connections=[
                Connection(Src0, Sw[0]),
                Connection(Src1, Sw[1]),
                Connection(Sw, Sco)
                ],
            events=[evt1, evt2],
            dt=0.01,
            log=False
            )

        Sim.run(duration=6.0, reset=True)

        time, [out] = Sco.read()

        #t < 2: output should be 1.0 (source 0)
        mask_0 = (time > 0.1) & (time < 1.9)
        self.assertTrue(np.allclose(out[mask_0], 1.0, atol=0.1),
            "Before switch: output should be 1.0")

        #2 < t < 4: output should be 5.0 (source 1)
        mask_1 = (time > 2.1) & (time < 3.9)
        self.assertTrue(np.allclose(out[mask_1], 5.0, atol=0.1),
            "After first switch: output should be 5.0")

        #t > 4: output should be 1.0 again (source 0)
        mask_2 = (time > 4.1) & (time < 5.9)
        self.assertTrue(np.allclose(out[mask_2], 1.0, atol=0.1),
            "After second switch: output should be 1.0")


    def test_switch_with_none_state(self):
        """Switch with None state should output 0"""

        Src = Constant(10.0)
        Sw = Switch(switch_state=None)
        Sco = Scope(labels=["output"])

        Sim = Simulation(
            blocks=[Src, Sw, Sco],
            connections=[
                Connection(Src, Sw[0]),
                Connection(Sw, Sco)
                ],
            dt=0.01,
            log=False
            )

        Sim.run(duration=1.0, reset=True)

        time, [out] = Sco.read()

        #with None state, output should be 0
        self.assertTrue(np.allclose(out, 0.0, atol=0.01))


class TestStateSpaceSystem(unittest.TestCase):
    """
    Test StateSpace block implementing a first-order system.

    System: dx/dt = A*x + B*u, y = C*x + D*u
    With A=-1, B=1, C=1, D=0 -> first order low-pass
    Step response: y(t) = 1 - exp(-t)
    """

    def setUp(self):

        Src = Source(lambda t: 1.0)  # step input

        #first-order system: dx/dt = -x + u, y = x
        self.SS = StateSpace(
            A=[[-1.0]],
            B=[[1.0]],
            C=[[1.0]],
            D=[[0.0]]
            )

        self.Sco = Scope(labels=["output"])

        self.Sim = Simulation(
            blocks=[Src, self.SS, self.Sco],
            connections=[
                Connection(Src, self.SS),
                Connection(self.SS, self.Sco)
                ],
            log=False
            )


    def test_step_response_explicit(self):
        """Verify step response with explicit adaptive solver"""

        self.Sim._set_solver(RKCK54, tolerance_lte_abs=1e-6, tolerance_lte_rel=0.0)
        self.Sim.run(duration=8, reset=True)

        time, [res] = self.Sco.read()
        ref = 1.0 - np.exp(-time)

        error = np.max(np.abs(ref - res))
        self.assertLess(error, 1e-4,
            f"Step response error: {error:.2e}")


    def test_step_response_implicit(self):
        """Verify step response with implicit adaptive solver"""

        self.Sim._set_solver(ESDIRK32, tolerance_lte_abs=1e-6, tolerance_lte_rel=0.0)
        self.Sim.run(duration=8, reset=True)

        time, [res] = self.Sco.read()
        ref = 1.0 - np.exp(-time)

        error = np.max(np.abs(ref - res))
        self.assertLess(error, 1e-4,
            f"Step response error: {error:.2e}")


class TestTransferFunctionSystem(unittest.TestCase):
    """
    Test TransferFunction block in a feedback system.

    Transfer function: H(s) = 1/(s+1) which is equivalent to
    the first-order system dx/dt = -x + u, y = x.
    """

    def test_tf_step_response(self):
        """Verify transfer function step response"""

        Src = Source(lambda t: 1.0)

        #H(s) = 1/(s+1)
        TF = TransferFunctionNumDen(Num=[1.0], Den=[1.0, 1.0])

        Sco = Scope(labels=["output"])

        Sim = Simulation(
            blocks=[Src, TF, Sco],
            connections=[
                Connection(Src, TF),
                Connection(TF, Sco)
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-6,
            log=False
            )

        Sim.run(duration=8, reset=True)

        time, [res] = Sco.read()
        ref = 1.0 - np.exp(-time)

        error = np.max(np.abs(ref - res))
        self.assertLess(error, 1e-4,
            f"TF step response error: {error:.2e}")


    def test_tf_in_feedback(self):
        """Test transfer function in a negative feedback loop"""

        #plant: H(s) = 1/(s+1)
        #controller: proportional gain K=2
        #closed loop: y/r = K*H/(1+K*H) = 2/(s+3)
        #step response: y(t) = 2/3 * (1 - exp(-3t))

        Src = Source(lambda t: 1.0)
        Err = Adder("+-")
        K = Amplifier(2.0)
        Plant = TransferFunctionNumDen(Num=[1.0], Den=[1.0, 1.0])
        Sco = Scope(labels=["output"])

        Sim = Simulation(
            blocks=[Src, Err, K, Plant, Sco],
            connections=[
                Connection(Src, Err),
                Connection(Plant, Err[1], Sco),
                Connection(Err, K),
                Connection(K, Plant)
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-8,
            log=False
            )

        Sim.run(duration=5, reset=True)

        time, [res] = Sco.read()
        ref = (2.0/3.0) * (1.0 - np.exp(-3.0 * time))

        error = np.max(np.abs(ref - res))
        self.assertLess(error, 1e-4,
            f"Feedback TF error: {error:.2e}")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
