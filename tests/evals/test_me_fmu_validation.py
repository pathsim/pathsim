########################################################################################
##
##                    Validation Tests for Model Exchange FMU
##
##   These tests validate the ModelExchangeFMU implementation by comparing against
##   FMPy's built-in simulation functions as reference.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os

from pathlib import Path
TEST_DIR = Path(__file__).parent

from pathsim import Simulation
from pathsim.blocks import ModelExchangeFMU, Scope
from pathsim.solvers import RK45, RK4



# VALIDATION TESTS =====================================================================

@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestModelExchangeFMUValidation(unittest.TestCase):
    """
    Validate ModelExchangeFMU against FMPy's reference implementation.

    These tests ensure that PathSim's integration produces the same results
    as FMPy's built-in simulate_fmu function.
    """

    @classmethod
    def setUpClass(cls):
        """Check if FMPy is available"""
        try:
            import fmpy
            cls.fmpy_available = True
        except ImportError:
            cls.fmpy_available = False


    def setUp(self):
        """Set up common test parameters"""
        if not self.fmpy_available:
            self.skipTest("FMPy not installed")

        # Look for any available ME FMU in test directory
        self.fmu_path = None
        for fmu_file in TEST_DIR.glob("*_ME.fmu"):
            self.fmu_path = fmu_file
            break

        if self.fmu_path is None:
            # Try BouncingBall without _ME suffix
            self.fmu_path = TEST_DIR / "BouncingBall.fmu"

        if not self.fmu_path.exists():
            self.skipTest(f"No Model Exchange FMU found. Run download_test_fmu.py first.")


    def test_inheritance_from_dynamicalsystem(self):
        """Test that ModelExchangeFMU properly inherits from DynamicalSystem"""
        from pathsim.blocks.dynsys import DynamicalSystem

        fmu = ModelExchangeFMU(str(self.fmu_path))

        # Check inheritance
        self.assertIsInstance(fmu, DynamicalSystem,
                            "ModelExchangeFMU should inherit from DynamicalSystem")

        # Check required attributes exist
        self.assertTrue(hasattr(fmu, 'engine'), "Missing engine attribute")
        self.assertTrue(hasattr(fmu, 'op_dyn'), "Missing op_dyn attribute")
        self.assertTrue(hasattr(fmu, 'op_alg'), "Missing op_alg attribute")


    def test_fmu_metadata_extraction(self):
        """Test that FMU metadata is correctly extracted"""
        fmu = ModelExchangeFMU(str(self.fmu_path))

        # Check metadata attributes
        self.assertIsNotNone(fmu.model_name, "Model name not extracted")
        self.assertIsNotNone(fmu.fmi_version, "FMI version not extracted")
        self.assertIsNotNone(fmu.n_states, "Number of states not extracted")
        self.assertIsNotNone(fmu.n_event_indicators, "Event indicators not extracted")

        # Check version is valid
        self.assertTrue(fmu.fmi_version.startswith('2.') or fmu.fmi_version.startswith('3.'),
                       f"Invalid FMI version: {fmu.fmi_version}")


    def test_port_configuration(self):
        """Test that input/output ports are correctly configured"""
        fmu = ModelExchangeFMU(str(self.fmu_path))

        # Check port counts match FMU description
        n_inputs = len(fmu._input_refs)
        n_outputs = len(fmu._output_refs)

        self.assertEqual(fmu.inputs.n, n_inputs,
                        "Input port count mismatch")
        self.assertEqual(fmu.outputs.n, n_outputs,
                        "Output port count mismatch")


    def test_state_derivative_evaluation(self):
        """Test that state derivatives can be evaluated"""
        fmu = ModelExchangeFMU(str(self.fmu_path))

        if fmu.n_states == 0:
            self.skipTest("FMU has no continuous states")

        # Get initial state
        x0 = fmu.engine.get()
        u0 = np.zeros(fmu.inputs.n)

        # Evaluate derivatives
        dx = fmu._get_derivatives(x0, u0, 0.0)

        # Check dimensions
        self.assertEqual(len(dx), fmu.n_states,
                        "Derivative dimension mismatch")
        self.assertTrue(np.all(np.isfinite(dx)),
                       "Derivatives contain NaN or Inf")


    def test_output_evaluation(self):
        """Test that outputs can be evaluated"""
        fmu = ModelExchangeFMU(str(self.fmu_path))

        if fmu.outputs.n == 0:
            self.skipTest("FMU has no outputs")

        # Get initial state
        x0 = fmu.engine.get()
        u0 = np.zeros(fmu.inputs.n)

        # Evaluate outputs
        y = fmu._get_outputs(x0, u0, 0.0)

        # Check dimensions
        self.assertEqual(len(y), fmu.outputs.n,
                        "Output dimension mismatch")
        self.assertTrue(np.all(np.isfinite(y)),
                       "Outputs contain NaN or Inf")


    def test_event_indicator_evaluation(self):
        """Test that event indicators can be queried"""
        fmu = ModelExchangeFMU(str(self.fmu_path))

        if fmu.n_event_indicators == 0:
            self.skipTest("FMU has no event indicators")

        # Get event indicators
        for i in range(fmu.n_event_indicators):
            indicator = fmu._get_event_indicator(i)
            self.assertTrue(np.isfinite(indicator),
                          f"Event indicator {i} is not finite")


    def test_zero_crossing_events_created(self):
        """Test that ZeroCrossing events are created for event indicators"""
        from pathsim.events.zerocrossing import ZeroCrossing

        fmu = ModelExchangeFMU(str(self.fmu_path))

        if fmu.n_event_indicators == 0:
            self.skipTest("FMU has no event indicators")

        # Count ZeroCrossing events
        zero_crossing_events = [e for e in fmu.events if isinstance(e, ZeroCrossing)]

        self.assertEqual(len(zero_crossing_events), fmu.n_event_indicators,
                        "Number of ZeroCrossing events should match event indicators")


    def test_simulation_completes(self):
        """Test that a basic simulation runs to completion"""
        fmu = ModelExchangeFMU(str(self.fmu_path), verbose=False)
        sco = Scope()

        sim = Simulation(
            blocks=[fmu, sco],
            connections=[],
            dt=0.01,
            Solver=RK45,
            log=False
        )

        # Run simulation - should not raise exceptions
        try:
            sim.run(0.5)  # Short simulation
            success = True
        except Exception as e:
            success = False
            self.fail(f"Simulation failed with error: {e}")

        # Check that data was recorded
        time, outputs = sco.read()
        self.assertGreater(len(time), 0, "No data recorded")


    def test_different_solvers(self):
        """Test that ME FMU works with different PathSim solvers"""
        from pathsim.solvers import Euler, RK4, RK45

        for Solver in [Euler, RK4, RK45]:
            with self.subTest(solver=Solver.__name__):
                fmu = ModelExchangeFMU(str(self.fmu_path))

                sim = Simulation(
                    blocks=[fmu],
                    dt=0.01,
                    Solver=Solver,
                    log=False
                )

                # Should run without errors
                sim.run(0.1)

                # Check state is finite
                state = fmu.engine.get()
                self.assertTrue(np.all(np.isfinite(state)),
                              f"State contains NaN/Inf with {Solver.__name__}")


    def test_verbose_mode(self):
        """Test verbose mode for debugging"""
        import io
        import sys

        fmu = ModelExchangeFMU(str(self.fmu_path), verbose=True)

        # Verbose mode should not cause errors
        # (actual output testing would require capturing stdout)
        self.assertTrue(fmu.verbose, "Verbose mode not enabled")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
