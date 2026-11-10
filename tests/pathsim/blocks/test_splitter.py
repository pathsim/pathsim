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

from pathsim import Simulation, Connection, Carrier, Vector
from pathsim.blocks import Constant, Integrator
from pathsim.blocks._block import Block
from pathsim.blocks.splitter import Splitter


# TESTS ================================================================================

class Species(Carrier):
    keys = ("H", "D", "T")


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


    def test_carrier(self):

        S = Splitter([0.9, 0.1], carrier=Species())

        #ports with the layout of the carrier
        self.assertEqual(len(S.inputs), 3)
        self.assertEqual(len(S.outputs), 6)
        self.assertEqual(S.output_port_labels["out_1"].start, 3)
        self.assertEqual(S.output_port_labels["out_1"].keys, ("H", "D", "T"))

        S.inputs["in"] = [1.0, 2.0, 3.0]
        S.update(0)

        np.testing.assert_array_almost_equal(S.outputs["out_0"], [0.9, 1.8, 2.7])
        np.testing.assert_array_almost_equal(S.outputs["out_1"], [0.1, 0.2, 0.3])

        #single channel of an output
        self.assertAlmostEqual(S.outputs[S.output_port_labels["out_1"]["T"]], 0.3)


    def test_vector(self):

        S = Splitter([0.5, 0.5], carrier=Vector(2))

        S.inputs["in"] = [2.0, 4.0]
        S.update(0)

        np.testing.assert_array_almost_equal(S.outputs.to_array(), [1.0, 2.0, 1.0, 2.0])


    def test_simulation(self):
        """species are conserved over the split"""

        class Source(Block):
            output_port_labels = {"out": Species()}

            def update(self, t):
                self.outputs["out"] = [1.0, 2.0, 3.0]

        src = Source()
        spl = Splitter([0.7, 0.3], carrier=Species())
        itg_0, itg_1 = Integrator([0.0]*3), Integrator([0.0]*3)

        sim = Simulation(
            [src, spl, itg_0, itg_1],
            [
                Connection(src["out"], spl["in"]),
                Connection(spl["out_0"], itg_0[0:3]),
                Connection(spl["out_1"], itg_1[0:3]),
                ],
            dt=0.1,
            log=False
            )
        sim.run(1.0)

        np.testing.assert_array_almost_equal(itg_0.state + itg_1.state, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(itg_1.state, [0.3, 0.6, 0.9])


    def test_linearization(self):

        S = Splitter([0.25, 0.75], carrier=Species())

        A, B, C, D = S.to_statespace(0)

        np.testing.assert_array_almost_equal(
            D, np.vstack([0.25*np.eye(3), 0.75*np.eye(3)])
            )


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
