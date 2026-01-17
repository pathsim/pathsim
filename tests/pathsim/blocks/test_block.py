########################################################################################
##
##                                  TESTS FOR 
##                              'blocks._block.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks._block import Block
from pathsim.blocks.amplifier import Amplifier
from pathsim.utils.register import Register
from pathsim.utils.portreference import PortReference


# TESTS ================================================================================

class TestBlock(unittest.TestCase):
    """
    Test the implementation of the base 'Block' class
    """

    def test_init(self):

        B = Block()

        #test default inputs and outputs
        self.assertTrue(isinstance(B.inputs, Register))
        self.assertTrue(isinstance(B.outputs, Register))

        self.assertEqual(len(B.inputs), 1)
        self.assertEqual(len(B.outputs), 1)

        self.assertEqual(B.inputs[0], 0.0)
        self.assertEqual(B.outputs[0], 0.0)

        #test default engine not defined
        self.assertEqual(B.engine, None)

        #is active by default
        self.assertTrue(B._active)

        #operators not defined by default
        self.assertEqual(B.op_alg, None)
        self.assertEqual(B.op_dyn, None)


    def test_len(self):

        B = Block()

        #test default len method
        self.assertEqual(len(B), 1)
            

    def test_on_off_bool(self):
        
        B = Block()

        #default active
        self.assertTrue(B)

        #deactivate block
        B.off()
        self.assertFalse(B)

        #activate block
        B.on()
        self.assertTrue(B)


    def test_getitem(self):

        B = Block()

        #test default getitem method
        pr = B[0]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.block, B)
        self.assertEqual(pr.ports, [0])

        pr = B[2]
        self.assertEqual(pr.ports, [2])

        pr = B[30]
        self.assertEqual(pr.ports, [30])

        #test input validation
        with self.assertRaises(ValueError): B[0.2]
        with self.assertRaises(ValueError): B[1j]
        with self.assertRaises(ValueError): B["a"]


    def test_getitem_slice(self):

        B = Block()

        #test slicing in getitem
        pr = B[:1]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0])

        pr = B[:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 1])

        pr = B[1:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [1])

        pr = B[0:5]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 1, 2, 3, 4])

        pr = B[3:7]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [3, 4, 5, 6])

        pr = B[3:7:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [3, 5])

        pr = B[:10:3]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 3, 6, 9])

        pr = B[2:12:4]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [2, 6, 10])

        #slice input validation
        with self.assertRaises(ValueError): B[1:] #open ended
        with self.assertRaises(ValueError): B[:0] #starting at zero




    def test_reset(self):

        B = Block()

        B.inputs.update_from_array([1, 2, 3])
        B.outputs.update_from_array([-1, 5])


        #test if inputs and outputs are set correctly
        self.assertEqual(B.inputs[0], 1)
        self.assertEqual(B.inputs[1], 2)
        self.assertEqual(B.inputs[2], 3)

        self.assertEqual(B.outputs[0], -1)
        self.assertEqual(B.outputs[1], 5)


        B.reset()

        #test if inputs and outputs are reset correctly
        self.assertEqual(B.inputs[0], 0.0)
        self.assertEqual(B.inputs[1], 0.0)
        self.assertEqual(B.inputs[2], 0.0)

        self.assertEqual(B.outputs[0], 0.0)
        self.assertEqual(B.outputs[1], 0.0)


    def test_update(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.update(None), 0.0)


    def test_solve(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.solve(None, None), 0.0)


    def test_step(self):

        B = Block()

        #test default implementation (scale=None means no rescale needed)
        self.assertEqual(B.step(None, None), (True, 0.0, None))


    def test_info_base_block(self):
        """Test info method on base Block class"""

        info = Block.info()

        #check all expected keys are present
        expected_keys = {"type", "description", "input_port_labels", "output_port_labels", "parameters"}
        self.assertEqual(set(info.keys()), expected_keys)

        #check type is correct
        self.assertEqual(info["type"], "Block")

        #check description is the docstring
        self.assertIn("Base 'Block' object", info["description"])

        #check port labels (base Block has dynamic ports)
        self.assertIsNone(info["input_port_labels"])
        self.assertIsNone(info["output_port_labels"])

        #check parameters (base Block has no parameters)
        self.assertEqual(info["parameters"], {})


    def test_info_with_parameters(self):
        """Test info method on block with parameters"""

        info = Amplifier.info()

        #check type is correct
        self.assertEqual(info["type"], "Amplifier")

        #check description contains relevant info
        self.assertIn("Amplifies", info["description"])

        #check port labels (Amplifier has dynamic ports)
        self.assertIsNone(info["input_port_labels"])
        self.assertIsNone(info["output_port_labels"])

        #check parameters include gain with default
        self.assertIn("gain", info["parameters"])
        self.assertEqual(info["parameters"]["gain"]["default"], 1.0)


    def test_info_caching(self):
        """Test that info method results are cached"""

        #clear cache first
        Block.info.cache_clear()

        #call info twice
        info1 = Block.info()
        info2 = Block.info()

        #should be same object due to caching
        self.assertIs(info1, info2)

        #check cache was used
        cache_info = Block.info.cache_info()
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.misses, 1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)