########################################################################################
##
##                                  TESTS FOR
##                             'blocks.splitter.py'
##
##                              Milan Rother 2026
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Integrator
from pathsim.blocks.splitter import Splitter


# TESTS ================================================================================

class TestSplitter(unittest.TestCase):
    """
    Test the implementation of the 'Splitter' block class
    """

    def test_init(self):

        #default, single output
        S = Splitter()
        np.testing.assert_array_equal(S.fractions, [1.0])
        self.assertEqual(S.output_port_labels, {"out_0": 0})

        #multiple outputs
        S = Splitter([0.2, 0.3, 0.5])
        self.assertEqual(S.input_port_labels, {"in": 0})
        self.assertEqual(S.output_port_labels, {"out_0": 0, "out_1": 1, "out_2": 2})
        self.assertEqual(len(S.outputs), 3)

        #input validation
        with self.assertRaises(ValueError): Splitter([0.5, 0.6])


    def test_len(self):

        S = Splitter([0.5, 0.5])

        self.assertEqual(len(S), 1)


    def test_update(self):

        S = Splitter([0.2, 0.3, 0.5])

        S.inputs[0] = 2.0
        S.update(0)

        np.testing.assert_array_almost_equal(S.outputs.to_array(), [0.4, 0.6, 1.0])
        self.assertAlmostEqual(S.outputs["out_1"], 0.6)


    def test_simulation(self):
        """the flow is conserved over the split"""

        src = Constant(2.0)
        spl = Splitter([0.7, 0.3])
        itg_0, itg_1 = Integrator(), Integrator()

        sim = Simulation(
            [src, spl, itg_0, itg_1],
            [
                Connection(src, spl["in"]),
                Connection(spl["out_0"], itg_0),
                Connection(spl["out_1"], itg_1),
                ],
            dt=0.1,
            log=False
            )
        sim.run(1.0)

        self.assertAlmostEqual(float(itg_0.state[0] + itg_1.state[0]), 2.0)
        self.assertAlmostEqual(float(itg_1.state[0]), 0.6)


    def test_linearization(self):

        S = Splitter([0.25, 0.75])

        A, B, C, D = S.to_statespace(0)

        np.testing.assert_array_almost_equal(D, [[0.25], [0.75]])


    def test_mutable(self):

        S = Splitter([0.5, 0.5])

        S.fractions = [0.2, 0.3, 0.5]

        self.assertEqual(len(S.outputs), 3)

        S.inputs[0] = 1.0
        S.update(0)

        np.testing.assert_array_almost_equal(S.outputs.to_array(), [0.2, 0.3, 0.5])


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
