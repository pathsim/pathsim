########################################################################################
##
##                                  TESTS FOR
##                               'blocks.fmu.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.fmu import CoSimulationFMU, ModelExchangeFMU

from pathsim.solvers._solver import Solver


# TESTS ================================================================================

class TestCoSimulationFMU(unittest.TestCase):
    """
    Test the implementation of the 'CoSimulationFMU' block class

    Note: Most FMU tests require FMPy and actual FMU files, making comprehensive
    testing difficult without proper test fixtures.
    """

    def test_import_error_without_fmpy(self):
        """Test that ImportError is raised when FMPy is not available"""
        import sys
        import importlib

        # This test only works if FMPy is not installed
        # If FMPy is installed, we skip this test
        try:
            import fmpy
            self.skipTest("FMPy is installed, cannot test ImportError case")
        except ImportError:
            pass

        # Try to create FMU without FMPy installed
        with self.assertRaises(ImportError) as context:
            fmu = CoSimulationFMU("nonexistent.fmu")

        self.assertIn("FMPy", str(context.exception))


class TestModelExchangeFMU(unittest.TestCase):
    """
    Test the implementation of the 'ModelExchangeFMU' block class

    Note: Most FMU tests require FMPy and actual FMU files, making comprehensive
    testing difficult without proper test fixtures.
    """

    def test_import_error_without_fmpy(self):
        """Test that ImportError is raised when FMPy is not available"""
        import sys
        import importlib

        # This test only works if FMPy is not installed
        # If FMPy is installed, we skip this test
        try:
            import fmpy
            self.skipTest("FMPy is installed, cannot test ImportError case")
        except ImportError:
            pass

        # Try to create FMU without FMPy installed
        with self.assertRaises(ImportError) as context:
            fmu = ModelExchangeFMU("nonexistent.fmu")

        self.assertIn("FMPy", str(context.exception))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
