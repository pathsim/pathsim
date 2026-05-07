########################################################################################
##
##                                  TESTS FOR 
##                               'connection.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.portreference import PortReference
from pathsim.connection import Connection, Duplex
from pathsim.blocks._block import Block


# TESTS ================================================================================

class TestConnection(unittest.TestCase):
    """
    Test the implementation of the 'Connection' class
    """

    def test_init_none(self):
        
        #default
        with self.assertRaises(TypeError):
            C = Connection()


    def test_init_single(self):

        B1, B2 = Block(), Block()

        #default
        C = Connection(B1, B2)
        self.assertTrue(isinstance(C.source, PortReference))
        self.assertEqual(C.source.ports, [0])
        self.assertTrue(isinstance(C.targets, list))

        #mixed
        C = Connection(B1[0], B2)
        self.assertEqual(C.source.ports, [0])
        self.assertEqual(C.targets[0].ports, [0])

        C = Connection(B1[2], B2)
        self.assertEqual(C.source.ports, [2])

        with self.assertRaises(ValueError):
            C = Connection(B1[0:3], B2)
        
        #all
        C = Connection(B1[2], B2[9])
        self.assertEqual(C.source.ports, [2])
        self.assertEqual(C.targets[0].ports, [9])

        C = Connection(B1[:8:3], B2[1:6:2])
        self.assertEqual(C.source.ports, [0, 3, 6])
        self.assertEqual(C.targets[0].ports, [1, 3, 5])


    def test_init_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #default
        C = Connection(B1, B2, B3)
        self.assertTrue(isinstance(C.source, PortReference))
        self.assertEqual(C.source.ports, [0])
        self.assertTrue(isinstance(C.targets, list))
        self.assertEqual(len(C.targets), 2)
        self.assertEqual(C.targets[0].ports, [0])
        self.assertEqual(C.targets[1].ports, [0])

        #mixed
        C = Connection(B1[1], B2[3], B3)
        self.assertEqual(C.source.ports, [1])
        self.assertEqual(C.targets[0].ports, [3])
        self.assertEqual(C.targets[1].ports, [0])

        C = Connection(B1, B2[2], B3[0])
        self.assertEqual(C.source.ports, [0])
        self.assertEqual(C.targets[0].ports, [2])
        self.assertEqual(C.targets[1].ports, [0])

        #all
        C = Connection(B1[1:7], B3[3:9])
        self.assertEqual(C.source.ports, [1, 2, 3, 4, 5, 6])
        self.assertEqual(C.targets[0].ports, [3, 4, 5, 6, 7, 8])






    def test_update_single(self):

        B1, B2 = Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)

        C = Connection(B1, B2[3]) 
        C.update()
        self.assertEqual(B2.inputs[3], 3)

        C = Connection(B1[1], B2[3]) 
        B1.outputs[1] = 2.5
        C.update()
        self.assertEqual(B2.inputs[3], 2.5)


    def test_update_multi(self):

        B1, B2, B3 = Block(), Block(), Block()

        #test data transfer with default ports
        C = Connection(B1, B2, B3) 
        B1.outputs[0] = 3
        C.update()
        self.assertEqual(B2.inputs[0], 3)
        self.assertEqual(B3.inputs[0], 3)

        #test data transfer with specific ports
        C = Connection(B1, B2[3], B3[2]) 
        B1.outputs[0] = 55
        C.update()
        self.assertEqual(B2.inputs[3], 55)
        self.assertEqual(B3.inputs[2], 55)

        #test with sliced ports
        
        #test with tuple ports



    def test_on_off_bool(self):

        B1, B2 = Block(), Block()

        #default
        C = Connection(B1, B2)

        #default active
        self.assertTrue(C)

        #deactivate
        C.off()
        self.assertFalse(C)

        #activate
        C.on()
        self.assertTrue(C)


    def test_resolve_ports_grows_input_register(self):
        """resolve_ports must size the target input register so blocks
        accessing inputs by position before any connection.update() runs
        (e.g. inside an algebraic loop) see the correct number of slots.
        """
        B1, B2 = Block(), Block()

        #fresh block has size-1 input register from Block.__init__
        self.assertEqual(len(B2.inputs), 1)

        C = Connection(B1[2], B2[3])

        #ports validated but lazy resize not yet triggered
        self.assertEqual(len(B2.inputs), 1)
        self.assertEqual(len(B1.outputs), 1)

        C.resolve_ports()

        #both registers must accommodate the highest port index
        self.assertEqual(len(B2.inputs), 4)
        self.assertEqual(len(B1.outputs), 3)

        #new slots are zero-initialised
        self.assertEqual(B2.inputs[3], 0.0)


    def test_resolve_ports_is_idempotent(self):
        """Repeated calls must not change register size or contents."""
        B1, B2 = Block(), Block()
        C = Connection(B1[5], B2[7])

        C.resolve_ports()
        size_in = len(B2.inputs)
        size_out = len(B1.outputs)

        #seed a value so we can detect data loss on a second resolve
        B2.inputs[7] = 42.0

        C.resolve_ports()
        self.assertEqual(len(B2.inputs), size_in)
        self.assertEqual(len(B1.outputs), size_out)
        self.assertEqual(B2.inputs[7], 42.0)


    def test_resolve_ports_multi_target(self):
        """All targets must be resolved, not just the first."""
        B1, B2, B3 = Block(), Block(), Block()
        C = Connection(B1[1], B2[4], B3[6])

        C.resolve_ports()

        self.assertEqual(len(B1.outputs), 2)
        self.assertEqual(len(B2.inputs), 5)
        self.assertEqual(len(B3.inputs), 7)



class TestDuplex(unittest.TestCase):
    """
    Test the implementation of the 'Duplex' class (bidirectional connection)
    """

    def test_init_none(self):
        
        #default
        with self.assertRaises(TypeError):
            D = Duplex()


    def test_init_mixed(self):

        B1, B2 = Block(), Block()

        #default

        D = Duplex(B1, B2)
        self.assertTrue(isinstance(D.source, PortReference))
        self.assertTrue(isinstance(D.target, PortReference))
        self.assertTrue(isinstance(D.targets, list))
        self.assertEqual(len(D.targets), 2)
        self.assertEqual(D.source.ports, [0])
        self.assertEqual(D.target.ports, [0])

        #specific

        D = Duplex(B1[3], B2)
        self.assertEqual(D.source.ports, [3])
        self.assertEqual(D.target.ports, [0])

        D = Duplex(B1, B2[2])
        self.assertEqual(D.source.ports, [0])
        self.assertEqual(D.target.ports, [2])

        D = Duplex(B1[5], B2[1])
        self.assertEqual(D.source.ports, [5])
        self.assertEqual(D.target.ports, [1])

        #slicing
        
        D = Duplex(B1[1:4], B2[1:4])
        self.assertEqual(D.source.ports, [1, 2, 3])
        self.assertEqual(D.target.ports, [1, 2, 3])
        
        D = Duplex(B1[:3], B2[1:4])
        self.assertEqual(D.source.ports, [0, 1, 2])
        self.assertEqual(D.target.ports, [1, 2, 3])

        #test too many

        B3 = Block()
        with self.assertRaises(TypeError): 
            D = Duplex(B1[3], B2, B3)


    def test_update(self):

        B1, B2 = Block(), Block()

        #default

        D = Duplex(B1, B2) 
        B1.outputs[0] = 3
        B2.outputs[0] = 1
        D.update()
        self.assertEqual(B1.inputs[0], 1)
        self.assertEqual(B2.inputs[0], 3)

        #specific

        D = Duplex(B1[3], B2[1]) 
        B1.outputs[3] = 2
        B2.outputs[1] = -0.1
        D.update()
        self.assertEqual(B1.inputs[3], -0.1)
        self.assertEqual(B2.inputs[1], 2)

        #slicing

        D = Duplex(B1[1:4], B2[:3])
        B1.outputs[0] = 33
        B1.outputs[1] = 99
        B1.outputs[2] = 44
        B1.outputs[3] = 77
        B1.outputs[4] = 11
        B2.outputs[0] = 0.33
        B2.outputs[1] = 0.99
        B2.outputs[2] = 0.44
        B2.outputs[3] = 0.77
        B2.outputs[4] = 0.11
        D.update()
        self.assertEqual(B1.inputs[3], 0.44)
        self.assertEqual(B2.inputs[1], 44)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)