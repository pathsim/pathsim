########################################################################################
##
##                        Testing System with Model Exchange FMU
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os

from pathlib import Path
TEST_DIR = Path(__file__).parent

from pathsim import Simulation, Connection
from pathsim.blocks import Scope, Source, ModelExchangeFMU
from pathsim.solvers import RK45



# TESTCASE =============================================================================

@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestModelExchangeFMUSystem(unittest.TestCase):
    """
    Test Model Exchange FMU integration with PathSim simulation.

    Note: This test requires a Model Exchange FMU file. Common test FMUs:
    - BouncingBall (has state events - ball bouncing)
    - Dahlquist (simple ODE test)
    - VanDerPol (stiff ODE)

    You can download reference FMUs from:
    https://github.com/modelica/fmi-cross-check
    """

    def setUp(self):
        """Set up test with BouncingBall FMU if available"""

        # Check for BouncingBall ME FMU in test directory
        self.fmu_path = TEST_DIR / "BouncingBall_ME.fmu"

        if not self.fmu_path.exists():
            self.skipTest(f"Model Exchange FMU not found at {self.fmu_path}")

        # Create FMU block
        self.fmu = ModelExchangeFMU(
            str(self.fmu_path),
            tolerance=1e-8,
            verbose=False
        )

        # Create scope to record outputs
        self.sco = Scope()

        # Initialize simulation
        self.sim = Simulation(
            blocks=[self.fmu, self.sco],
            connections=[
                Connection(self.fmu[:], self.sco[:])
            ],
            dt=0.01,
            Solver=RK45,
            log=False
        )


    def test_basic_simulation(self):
        """Test basic ME FMU simulation runs without errors"""

        # Run simulation
        self.sim.run(1.0)

        # Read scope data
        time, outputs = self.sco.read()

        # Basic checks
        self.assertGreater(len(time), 0, "No simulation data recorded")
        self.assertEqual(len(outputs), self.fmu.outputs.n, "Output dimension mismatch")


    def test_state_events(self):
        """Test that state events (zero-crossings) are detected and handled"""

        # For BouncingBall, we expect multiple bounce events
        if "BouncingBall" not in str(self.fmu_path):
            self.skipTest("State event test requires BouncingBall FMU")

        # Run simulation
        self.sim.run(3.0)

        # Check that events were detected
        n_events = 0
        for event in self.fmu.events:
            if hasattr(event, '_times'):
                n_events += len(event._times)

        # BouncingBall should have multiple bounces (state events)
        self.assertGreater(n_events, 0, "No state events detected")


    def test_continuous_states(self):
        """Test that FMU continuous states are properly integrated"""

        # Run simulation
        self.sim.run(1.0)

        # Get final state from solver
        final_state = self.fmu.engine.get()

        # Check state dimension matches FMU
        self.assertEqual(len(final_state), self.fmu.n_states,
                        f"State dimension mismatch: {len(final_state)} vs {self.fmu.n_states}")


    def test_with_inputs(self):
        """Test ME FMU with external inputs"""

        # Only run if FMU has inputs
        if len(self.fmu._input_refs) == 0:
            self.skipTest("FMU has no inputs")

        # Create input source
        src = Source(lambda t: np.sin(2*np.pi*t))

        # Recreate simulation with input
        sim = Simulation(
            blocks=[src, self.fmu, self.sco],
            connections=[
                Connection(src[0], self.fmu[0]),
                Connection(self.fmu[:], self.sco[:])
            ],
            dt=0.01,
            Solver=RK45,
            log=False
        )

        # Run simulation
        sim.run(1.0)

        # Read data
        time, outputs = self.sco.read()

        # Check simulation completed
        self.assertGreater(len(time), 0, "No simulation data with inputs")


    def test_reset(self):
        """Test FMU reset functionality"""

        # Run simulation
        self.sim.run(1.0)
        state_after_run = self.fmu.engine.get().copy()

        # Reset FMU
        self.fmu.reset()
        state_after_reset = self.fmu.engine.get()

        # State should be different after reset (back to initial)
        # Unless the final state happens to equal initial state
        initial_state = self.fmu.initial_value

        np.testing.assert_array_almost_equal(
            state_after_reset,
            initial_state,
            err_msg="Reset did not restore initial state"
        )


    def test_time_events(self):
        """Test FMU time events if supported"""

        # Run simulation
        self.sim.run(2.0)

        # Check if time events were created
        if self.fmu.time_event is not None:
            self.assertIsNotNone(self.fmu.time_event.times_evt,
                               "Time event list not initialized")
            self.assertGreater(len(self.fmu.time_event.times_evt), 0,
                             "No time events scheduled")


    def test_event_indicators(self):
        """Test that event indicators can be queried"""

        if self.fmu.n_event_indicators == 0:
            self.skipTest("FMU has no event indicators")

        # Get event indicators at t=0
        indicators = self.fmu.fmu.getEventIndicators()

        # Check dimension
        self.assertEqual(len(indicators), self.fmu.n_event_indicators,
                        "Event indicator dimension mismatch")


# SIMPLE ANALYTICAL TEST ===============================================================

class TestModelExchangeFMUAnalytical(unittest.TestCase):
    """
    Test ME FMU against analytical solutions using simple test cases.

    This test creates a simple exponential decay FMU programmatically if possible,
    or uses a reference FMU with known analytical solution.
    """

    def test_exponential_decay(self):
        """Test simple exponential decay: dx/dt = -x, x(0) = 1"""

        # This test would require a simple exponential decay FMU
        # Skip for now until we have appropriate test FMUs
        self.skipTest("Requires simple exponential decay ME FMU")

        # Expected implementation:
        # fmu = ModelExchangeFMU("exponential_decay_ME.fmu")
        # sim = Simulation(blocks=[fmu], dt=0.01, Solver=RK45)
        # sim.run(5.0)
        #
        # # Analytical solution: x(t) = exp(-t)
        # time = sim.t
        # analytical = np.exp(-time)
        # numerical = fmu.engine.get()[0]
        #
        # np.testing.assert_allclose(numerical, analytical, rtol=1e-5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
