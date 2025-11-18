########################################################################################
##
##                    Testing ModelExchangeFMU analytical validation
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import ModelExchangeFMU, DynamicalSystem, Scope
from pathsim.solvers import RKDP54


# TESTCASE =============================================================================

class TestModelExchangeFMUAnalytical(unittest.TestCase):

    def test_structure(self):
        """Test that ModelExchangeFMU has the correct structure and inheritance."""

        # Check inheritance
        self.assertTrue(issubclass(ModelExchangeFMU, DynamicalSystem))

        # Check required methods exist
        required_methods = [
            '_get_derivatives',
            '_get_outputs',
            '_handle_event',
            '_update_time_events',
            'sample',
            'reset'
        ]

        for method in required_methods:
            self.assertTrue(
                hasattr(ModelExchangeFMU, method),
                f"ModelExchangeFMU missing required method: {method}"
            )


    def test_dynamical_system_integration(self):
        """Test integration of a simple ODE system to validate numerical accuracy.

        Uses exponential decay: dx/dt = -x, x(0) = 1
        Analytical solution: x(t) = exp(-t)
        """

        k = 1.0
        x0 = 1.0

        # Create simple dynamical system
        sys = DynamicalSystem(
            func_dyn=lambda x, u, t: -k * x,
            func_alg=lambda x, u, t: x,
            initial_value=x0
        )

        sco = Scope()

        sim = Simulation(
            blocks=[sys, sco],
            connections=[Connection(sys[0], sco[0])],
            dt=0.01,
            Solver=RKDP54,
            log=False
        )

        t_final = 5.0
        sim.run(t_final)

        time, outputs = sco.read()
        x_numerical = outputs[0]

        # Analytical solution
        x_analytical = x0 * np.exp(-k * time)

        # Calculate error
        error = np.abs(x_numerical - x_analytical)
        max_error = np.max(error)

        # Check if error is acceptable
        tolerance = 1e-4
        self.assertLess(
            max_error,
            tolerance,
            f"Integration error {max_error:.2e} exceeds tolerance {tolerance}"
        )


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
