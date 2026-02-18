########################################################################################
##
##                                  TESTS FOR 
##                              'blocks.switch.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.switch import Switch


# TESTS ================================================================================

class TestSwitch(unittest.TestCase):
    """
    Test the implementation of the 'Switch' block class
    """

    def test_init(self):

        #test default initialization
        S = Switch()
        self.assertEqual(S.switch_state, None)

        #test special initialization
        S = Switch(1)
        self.assertEqual(S.switch_state, 1)

        S = Switch(switch_state=0)
        self.assertEqual(S.switch_state, 0)


    def test_len(self):

        #has no direct passthrough
        S = Switch()
        self.assertEqual(len(S), 0)

        #has direct passthrough
        S = Switch(0)
        self.assertEqual(len(S), 1)


    def test_select(self):

        #test the switch state selector
        S = Switch()
        self.assertEqual(S.switch_state, None)

        S.select(0)
        self.assertEqual(S.switch_state, 0)

        S.select(1)
        self.assertEqual(S.switch_state, 1)

        S.select(3)
        self.assertEqual(S.switch_state, 3)


    def test_update(self):

        #test default initialization
        S = Switch()
        self.assertEqual(S.switch_state, None)

        S.inputs[0] = 3
        S.update(0)

        #test if no passthrough
        self.assertEqual(S.outputs[0], 0.0)

        #test switch setting
        S = Switch(3)
        self.assertEqual(S.switch_state, 3)

        S.inputs[0] = 3
        S.inputs[1] = 4
        S.inputs[2] = 5
        S.inputs[3] = 6
        S.inputs[4] = 7

        S.update(0)

        self.assertEqual(S.outputs[0], 6)

        S.select(1)
        S.update(0)

        self.assertEqual(S.outputs[0], 4)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
