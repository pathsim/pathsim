########################################################################################
##
##                                  TESTS FOR 
##                              'blocks.delay.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.delay import Delay
from pathsim.utils.adaptivebuffer import AdaptiveBuffer

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestDelay(unittest.TestCase):
    """
    Test the implementation of the 'Delay' block class
    """

    def test_init(self):

        #test specific initialization
        D = Delay(tau=1)

        self.assertTrue(isinstance(D._buffer, AdaptiveBuffer))
        self.assertEqual(D.tau, 1)


    def test_embedding(self):

        D = Delay(tau=10)
        E = Embedding(D, np.sin, lambda t: np.sin(t-10) if t>10 else 0.0)
        for t in range(100): self.assertEqual(*E.check_SISO(t))


    def test_len(self):
        
        D = Delay()

        #no passthrough
        self.assertEqual(len(D), 0)


    def test_reset(self):

        D = Delay(tau=100)

        for t in range(10):
            D.sample(t, 1.0)

        self.assertEqual(len(D._buffer), 10)

        D.reset()

        #test if reset worked
        self.assertEqual(len(D._buffer), 0)  
        

    def test_sample(self):

        D = Delay(tau=100)

        for t in range(10):

            #test internal buffer length
            self.assertEqual(len(D._buffer), t)

            D.sample(t, None)


    def test_update(self):

        #test delay without interpolation
        D = Delay(tau=10)

        for t in range(100):

            D.inputs[0] = t
            D.sample(t, None)

            D.update(t)

            #test if delay is correctly applied
            self.assertEqual(D.outputs[0], max(0, t-10))

        #test delay with local interpolation
        D = Delay(tau=10.5)

        for t in range(100):

            D.inputs[0] = t
            D.sample(t, None)

            D.update(t)

            #test if delay is correctly applied
            self.assertEqual(D.outputs[0], max(0, t-10.5))


class TestDelayDiscrete(unittest.TestCase):
    """
    Test the discrete-time (sampling_period) mode of the 'Delay' block class
    """

    def test_init_discrete(self):

        D = Delay(tau=0.01, sampling_period=0.001)

        self.assertEqual(D._n, 10)
        self.assertEqual(len(D._ring), 10)
        self.assertTrue(hasattr(D, 'events'))
        self.assertEqual(len(D.events), 1)


    def test_n_computation(self):

        #exact multiple
        D = Delay(tau=0.05, sampling_period=0.01)
        self.assertEqual(D._n, 5)

        #rounding
        D = Delay(tau=0.015, sampling_period=0.01)
        self.assertEqual(D._n, 2)

        #minimum of 1
        D = Delay(tau=0.001, sampling_period=0.01)
        self.assertEqual(D._n, 1)


    def test_len(self):

        D = Delay(tau=0.01, sampling_period=0.001)

        #no passthrough
        self.assertEqual(len(D), 0)


    def test_reset(self):

        D = Delay(tau=0.01, sampling_period=0.001)

        #push some values
        D._sample_next_timestep = True
        D.inputs[0] = 42.0
        D.sample(0, 0.001)

        D.reset()

        #ring buffer should be all zeros
        self.assertTrue(all(v == 0.0 for v in D._ring))
        self.assertEqual(len(D._ring), D._n)


    def test_discrete_delay(self):

        n = 3
        D = Delay(tau=0.003, sampling_period=0.001)

        self.assertEqual(D._n, n)

        #push values through the ring buffer
        outputs = []
        for k in range(10):
            D.inputs[0] = float(k)
            D._sample_next_timestep = True
            D.sample(k * 0.001, 0.001)
            D.update(k * 0.001)
            outputs.append(D.outputs[0])

        #first n outputs should be 0 (initial buffer fill)
        for k in range(n):
            self.assertEqual(outputs[k], 0.0, f"output[{k}] should be 0.0")

        #after that, output should be delayed by n samples
        for k in range(n, 10):
            self.assertEqual(outputs[k], float(k - n), f"output[{k}] should be {k-n}")


    def test_no_sample_without_flag(self):

        D = Delay(tau=0.003, sampling_period=0.001)

        #push a value without the flag set
        D.inputs[0] = 42.0
        D.sample(0, 0.001)

        #ring buffer should be unchanged (all zeros)
        self.assertTrue(all(v == 0.0 for v in D._ring))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)