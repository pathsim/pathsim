########################################################################################
##
##                    Testing relay-controlled thermostat system
##
##   Thermal plant with relay hysteresis controller. Verifies event-driven
##   switching behavior produces correct temperature oscillation pattern.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Integrator, Amplifier, Adder, Constant, Relay, Scope

from pathsim.solvers import (
    RKBS32, RKCK54, RKDP54, RKV65, RKDP87,
    ESDIRK32, ESDIRK43, ESDIRK54
    )


# TESTCASE =============================================================================

class TestRelayThermostatSystem(unittest.TestCase):
    """
    Thermostat system: relay controller with hysteresis driving a first-order
    thermal plant.

    System:
        heater = Relay(threshold_up=22, threshold_down=18, value_up=0, value_down=50)
        dT/dt = -alpha*(T - T_ambient) + heater_output / C

    When temperature rises above 22 -> heater OFF (value_up=0)
    When temperature drops below 18 -> heater ON (value_down=50)

    The system should oscillate between the two thresholds.
    """

    def setUp(self):

        #thermal parameters
        self.alpha = 0.5  # heat loss coefficient
        self.T_amb = 10.0  # ambient temperature
        self.C = 5.0  # thermal capacity

        #initial temperature (between thresholds)
        self.T0 = 20.0

        #blocks
        self.Int = Integrator(self.T0)  # temperature state
        Amp = Amplifier(-self.alpha)  # heat loss: -alpha * T
        Amb = Constant(self.alpha * self.T_amb)  # ambient contribution: alpha * T_amb
        Htr = Amplifier(1.0 / self.C)  # heater gain: heater / C
        Add = Adder()  # sum: -alpha*T + alpha*T_amb + heater/C

        self.Rly = Relay(
            threshold_up=22.0,
            threshold_down=18.0,
            value_up=0.0,     # heater off when T > 22
            value_down=50.0   # heater on when T < 18
            )

        self.Sco = Scope(labels=["temperature", "heater"])

        blocks = [self.Int, Amp, Amb, Htr, Add, self.Rly, self.Sco]

        #connections: T -> Amp, Amp -> Add[0], Amb -> Add[1], Rly -> Htr -> Add[2], Add -> Int
        connections = [
            Connection(self.Int, Amp, self.Rly, self.Sco[0]),
            Connection(Amp, Add[0]),
            Connection(Amb, Add[1]),
            Connection(self.Rly, Htr, self.Sco[1]),
            Connection(Htr, Add[2]),
            Connection(Add, self.Int)
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_thermostat_oscillation(self):
        """Test that temperature oscillates between thresholds"""

        self.Sim.run(duration=30, reset=True)

        time, [temp, heater] = self.Sco.read()

        #after initial transient (t>5), temperature should stay within bounds
        mask = time > 5
        temp_steady = temp[mask]

        #temperature should oscillate within reasonable bounds around thresholds
        self.assertTrue(np.min(temp_steady) > 16.0,
            f"Temperature dropped too low: {np.min(temp_steady):.2f}")
        self.assertTrue(np.max(temp_steady) < 24.0,
            f"Temperature rose too high: {np.max(temp_steady):.2f}")

        #heater should have switched multiple times
        heater_steady = heater[mask]
        switches = np.sum(np.abs(np.diff(heater_steady)) > 1)
        self.assertGreater(switches, 2, "Heater should have switched multiple times")


    def test_thermostat_with_adaptive_solvers(self):
        """Test thermostat with different adaptive solvers"""

        for SOL in [RKBS32, RKCK54, RKDP87]:

            with self.subTest(SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(duration=20, reset=True)

                time, [temp, _] = self.Sco.read()

                #temperature should stay bounded
                mask = time > 5
                self.assertTrue(np.min(temp[mask]) > 16.0)
                self.assertTrue(np.max(temp[mask]) < 24.0)


    def test_thermostat_with_implicit_solvers(self):
        """Test thermostat with implicit adaptive solvers"""

        for SOL in [ESDIRK32, ESDIRK43]:

            with self.subTest(SOL=str(SOL)):

                self.Sim.reset()
                self.Sim._set_solver(SOL, tolerance_lte_abs=1e-6)
                self.Sim.run(duration=20, reset=True)

                time, [temp, _] = self.Sco.read()

                #temperature should stay bounded
                mask = time > 5
                self.assertTrue(np.min(temp[mask]) > 16.0)
                self.assertTrue(np.max(temp[mask]) < 24.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
