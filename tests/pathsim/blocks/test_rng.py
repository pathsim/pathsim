########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.rng.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.rng import RandomNumberGenerator


# TESTS ================================================================================

class TestRandomNumberGenerator(unittest.TestCase):
    """
    Test the implementation of the 'RandomNumberGenerator' block class
    """

    def test_init(self):

        R = RandomNumberGenerator()

        self.assertEqual(R.sampling_period, None)
        self.assertEqual(R.events, [])

        R = RandomNumberGenerator(sampling_period=1)

        self.assertEqual(R.sampling_period, 1)
        self.assertEqual(R.events[0].t_period, R.sampling_period)


    def test_len(self):
        
        R = RandomNumberGenerator()

        #no passthrough
        self.assertEqual(len(R), 0)


    def test_reset(self):

        R = RandomNumberGenerator()

        for t in range(10):
            R.sample(t, None)

        R.reset()

        #test if reset worked
        self.assertEqual(R.outputs[0], 0.0)


    def test_sample(self):

        #first test default 'sampling_period=None'
        R = RandomNumberGenerator()

        for t in range(10):

            #test sample counter
            old = R.outputs[0]

            R.sample(t, None)
            R.update(t)

            #test if new random value is sampled
            self.assertNotEqual(old, R.outputs[0])

                

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)