########################################################################################
##
##                                  TESTS FOR
##                              'utils.carrier.py'
##
##                              Milan Rother 2026
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Connection
from pathsim.blocks._block import Block
from pathsim.utils.carrier import Carrier, Vector


# TESTS ================================================================================

class TestCarrier(unittest.TestCase):
    """
    test the 'Carrier' class
    """

    def test_init(self):
        #default
        C = Carrier()
        self.assertEqual(C.keys, ())
        self.assertEqual(C.start, 0)
        self.assertEqual(C.stop, 0)

        #with keys
        C = Carrier(keys=("F", "T", "P"))
        self.assertEqual(C.keys, ("F", "T", "P"))
        self.assertEqual(C.start, 0)
        self.assertEqual(C.stop, 3)

        #placed at an offset
        C = Carrier(5, keys=("F", "T"))
        self.assertEqual(C.start, 5)
        self.assertEqual(C.stop, 7)

        #input validation
        with self.assertRaises(ValueError): Carrier(keys=("F", "F"))     #duplicates
        with self.assertRaises(ValueError): Carrier(keys=("F", 1))       #no str
        with self.assertRaises(ValueError): Carrier(keys=("F", "start")) #reserved
        with self.assertRaises(ValueError): Carrier(keys=("F", "_x"))    #reserved
        with self.assertRaises(ValueError): Carrier(-1, keys=("F",))     #negative start
        with self.assertRaises(ValueError): Carrier(1.5, keys=("F",))    #no int


    def test_subclass(self):
        #keys from the class definition
        class Mixture(Carrier):
            keys = ("F", "T", "P")

        C = Mixture(2)
        self.assertEqual(C.keys, ("F", "T", "P"))
        self.assertEqual(C.F, 2)
        self.assertEqual(C.P, 4)

        #keys from the constructor
        class Stream(Carrier):
            def __init__(self, species, start=0):
                super().__init__(start, keys=("F", "T", *species))
                self.x = slice(self.T + 1, self.stop)

        C = Stream(("a", "b"), start=1)
        self.assertEqual(C.keys, ("F", "T", "a", "b"))
        self.assertEqual(C.F, 1)
        self.assertEqual(C.b, 4)
        self.assertEqual(C.x, slice(3, 5))


    def test_attributes(self):
        C = Carrier(3, keys=("F", "T", "P"))

        #channel indices as plain attributes
        self.assertEqual(C.F, 3)
        self.assertEqual(C.T, 4)
        self.assertEqual(C.P, 5)


    def test_len(self):
        self.assertEqual(len(Carrier()), 0)
        self.assertEqual(len(Carrier(keys=("F", "T"))), 2)
        self.assertEqual(len(Carrier(9, keys=("F", "T"))), 2)


    def test_iter(self):
        C = Carrier(2, keys=("F", "T", "P"))
        self.assertEqual(list(C), [2, 3, 4])


    def test_getitem(self):
        C = Carrier(2, keys=("F", "T", "P"))

        #by key
        self.assertEqual(C["F"], 2)
        self.assertEqual(C["P"], 4)

        #by position
        self.assertEqual(C[0], 2)
        self.assertEqual(C[2], 4)

        #input validation
        with self.assertRaises(ValueError): C["x"]  #no such key
        with self.assertRaises(ValueError): C[3]    #outside
        with self.assertRaises(ValueError): C[-1]   #outside
        with self.assertRaises(ValueError): C[1.5]  #no int, str


    def test_slice(self):
        C = Carrier(2, keys=("F", "T", "P"))
        self.assertEqual(C.slice, slice(2, 5))

        #direct access to flat data
        u = np.arange(6, dtype=float)
        np.testing.assert_array_equal(u[C.slice], [2.0, 3.0, 4.0])


    def test_to_array(self):
        C = Carrier(4, keys=("F", "T", "P"))

        np.testing.assert_array_equal(C.to_array({"F": 1.0, "T": 2.0, "P": 3.0}), [1.0, 2.0, 3.0])

        #missing keys stay zero
        np.testing.assert_array_equal(C.to_array({"T": 2.0}), [0.0, 2.0, 0.0])


class TestVector(unittest.TestCase):
    """
    test the 'Vector' class
    """

    def test_init(self):
        V = Vector(3)
        self.assertEqual(V.keys, ())
        self.assertEqual(V.start, 0)
        self.assertEqual(V.stop, 3)

        V = Vector(3, start=2)
        self.assertEqual(V.start, 2)
        self.assertEqual(V.stop, 5)

        #input validation
        with self.assertRaises(ValueError): Vector(0)   #not positive
        with self.assertRaises(ValueError): Vector(2.5) #no int


    def test_getitem(self):
        V = Vector(3, start=2)
        self.assertEqual(V[0], 2)
        self.assertEqual(V[2], 4)

        #input validation
        with self.assertRaises(ValueError): V[3]   #outside
        with self.assertRaises(ValueError): V["x"] #no keys


    def test_len_iter(self):
        V = Vector(3, start=2)
        self.assertEqual(len(V), 3)
        self.assertEqual(list(V), [2, 3, 4])


class TestCarrierPorts(unittest.TestCase):
    """
    test blocks that declare carriers as port labels
    """

    def make_blocks(self):

        class Source(Block):
            output_port_labels = {"out": Carrier(keys=("F", "T", "P"))}

        class Sink(Block):
            input_port_labels = {"in": Carrier(keys=("F", "T", "P")), "Q": 3}

        return Source(), Sink()


    def test_register_size(self):
        src, snk = self.make_blocks()

        #registers are pre-sized to hold all channels
        self.assertEqual(len(src.outputs), 3)
        self.assertEqual(len(snk.inputs), 4)


    def test_register_access(self):
        src, _ = self.make_blocks()

        #set and get all channels of the carrier at once
        src.outputs["out"] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(src.outputs["out"], [1.0, 2.0, 3.0])

        #single channels stay addressable
        self.assertEqual(src.outputs[1], 2.0)


    def test_portreference(self):
        src, snk = self.make_blocks()

        #a carrier port counts with all its channels
        PR = src["out"]
        self.assertEqual(len(PR), 3)
        np.testing.assert_array_equal(PR._get_output_indices(), [0, 1, 2])

        #plain channels are unaffected
        self.assertEqual(len(snk["Q"]), 1)


    def test_component_access(self):
        src, _ = self.make_blocks()

        #by attribute and by key
        self.assertEqual(src["out"].T.ports, [1])
        self.assertEqual(src["out"]["T"].ports, [1])
        self.assertEqual(src["out"][2].ports, [2])

        #only for a single carrier port
        with self.assertRaises(ValueError): src[0]["T"]


    def test_connection(self):
        src, snk = self.make_blocks()

        #whole carrier
        C = Connection(src["out"], snk["in"])
        self.assertEqual(len(C), 3)

        src.outputs["out"] = [1.0, 2.0, 3.0]
        C.update()
        np.testing.assert_array_equal(snk.inputs["in"], [1.0, 2.0, 3.0])

        #single component
        C = Connection(src["out"].P, snk["Q"])
        self.assertEqual(len(C), 1)

        C.update()
        self.assertEqual(snk.inputs["Q"], 3.0)


    def test_connection_dimensions(self):
        src, snk = self.make_blocks()

        #carrier width has to match
        with self.assertRaises(ValueError): Connection(src["out"], snk["Q"])
        with self.assertRaises(ValueError): Connection(src["out"].F, snk["in"])


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(buffer=True)
