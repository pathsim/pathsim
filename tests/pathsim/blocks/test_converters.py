########################################################################################
##
##                                  TESTS FOR
##                            'blocks.converters.py'
##
##                            Milan Rother 2024/25
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.converters import DAC, ADC
from pathsim.events.schedule import Schedule


# TESTS ================================================================================

class TestADC(unittest.TestCase):
    """
    Test the implementation of the base 'ADC' class
    """

    def test_init(self):
        """Test ADC initialization"""

        adc = ADC()

        self.assertTrue(isinstance(adc.events[0], Schedule))
        self.assertEqual(adc.n_bits, 4)
        self.assertEqual(adc.span, [-1, 1])
        self.assertEqual(adc.T, 1)
        self.assertEqual(adc.tau, 0)

        # Test custom initialization
        adc = ADC(n_bits=8, span=[0, 5], T=0.1, tau=0.05)

        self.assertEqual(adc.n_bits, 8)
        self.assertEqual(adc.span, [0, 5])
        self.assertEqual(adc.T, 0.1)
        self.assertEqual(adc.tau, 0.05)


    def test_len(self):
        """Test that ADC has no direct passthrough"""

        adc = ADC()

        self.assertEqual(len(adc), 0)


    def test_output_ports(self):
        """Test that ADC has correct number of output ports after sampling"""

        adc = ADC(n_bits=4)
        adc.events[0].func_act(0)  # trigger sampling to expand outputs
        self.assertEqual(len(adc.outputs), 4)

        adc = ADC(n_bits=8)
        adc.events[0].func_act(0)  # trigger sampling to expand outputs
        self.assertEqual(len(adc.outputs), 8)


    def test_sample_midrange(self):
        """Test ADC sampling at midrange value"""

        adc = ADC(n_bits=4, span=[-1, 1])
        adc.inputs[0] = 0.0  # Midrange

        # Trigger sampling
        adc.events[0].func_act(0)

        # For 4-bit ADC with span [-1,1]:
        # 0.0 maps to middle of range
        # scaled = (0 - (-1)) / 2 = 0.5
        # int_val = floor(0.5 * 16) = 8
        # binary: 1000 (MSB to LSB)
        # outputs: [0,0,0,1] (LSB to MSB)

        self.assertEqual(adc.outputs[0], 0)  # LSB
        self.assertEqual(adc.outputs[1], 0)
        self.assertEqual(adc.outputs[2], 0)
        self.assertEqual(adc.outputs[3], 1)  # MSB


    def test_sample_minimum(self):
        """Test ADC sampling at minimum value"""

        adc = ADC(n_bits=4, span=[-1, 1])
        adc.inputs[0] = -1.0  # Minimum

        # Trigger sampling
        adc.events[0].func_act(0)

        # -1.0 maps to 0
        # binary: 0000
        self.assertEqual(adc.outputs[0], 0)
        self.assertEqual(adc.outputs[1], 0)
        self.assertEqual(adc.outputs[2], 0)
        self.assertEqual(adc.outputs[3], 0)


    def test_sample_maximum(self):
        """Test ADC sampling at maximum value"""

        adc = ADC(n_bits=4, span=[-1, 1])
        adc.inputs[0] = 1.0  # Maximum

        # Trigger sampling
        adc.events[0].func_act(0)

        # 1.0 maps to max code (15)
        # binary: 1111
        self.assertEqual(adc.outputs[0], 1)  # LSB
        self.assertEqual(adc.outputs[1], 1)
        self.assertEqual(adc.outputs[2], 1)
        self.assertEqual(adc.outputs[3], 1)  # MSB


    def test_clipping_above(self):
        """Test ADC clipping when input exceeds maximum"""

        adc = ADC(n_bits=4, span=[-1, 1])
        adc.inputs[0] = 2.0  # Above maximum

        # Trigger sampling
        adc.events[0].func_act(0)

        # Should clip to 1.0, which maps to 1111
        self.assertEqual(adc.outputs[0], 1)
        self.assertEqual(adc.outputs[1], 1)
        self.assertEqual(adc.outputs[2], 1)
        self.assertEqual(adc.outputs[3], 1)


    def test_clipping_below(self):
        """Test ADC clipping when input below minimum"""

        adc = ADC(n_bits=4, span=[-1, 1])
        adc.inputs[0] = -2.0  # Below minimum

        # Trigger sampling
        adc.events[0].func_act(0)

        # Should clip to -1.0, which maps to 0000
        self.assertEqual(adc.outputs[0], 0)
        self.assertEqual(adc.outputs[1], 0)
        self.assertEqual(adc.outputs[2], 0)
        self.assertEqual(adc.outputs[3], 0)


    def test_different_span(self):
        """Test ADC with different input span"""

        adc = ADC(n_bits=2, span=[0, 10])
        adc.inputs[0] = 5.0  # Midrange

        # Trigger sampling
        adc.events[0].func_act(0)

        # scaled = (5 - 0) / 10 = 0.5
        # int_val = floor(0.5 * 4) = 2
        # binary: 10
        self.assertEqual(adc.outputs[0], 0)  # LSB
        self.assertEqual(adc.outputs[1], 1)  # MSB


class TestDAC(unittest.TestCase):
    """
    Test the implementation of the base 'DAC' class
    """

    def test_init(self):
        """Test DAC initialization"""

        dac = DAC()

        self.assertTrue(isinstance(dac.events[0], Schedule))
        self.assertEqual(dac.n_bits, 4)
        self.assertEqual(dac.span, [-1, 1])
        self.assertEqual(dac.T, 1)
        self.assertEqual(dac.tau, 0)

        # Test custom initialization
        dac = DAC(n_bits=8, span=[0, 5], T=0.1, tau=0.05)

        self.assertEqual(dac.n_bits, 8)
        self.assertEqual(dac.span, [0, 5])
        self.assertEqual(dac.T, 0.1)
        self.assertEqual(dac.tau, 0.05)


    def test_len(self):
        """Test that DAC has no direct passthrough"""

        dac = DAC()

        self.assertEqual(len(dac), 0)


    def test_input_ports(self):
        """Test that DAC has correct number of input ports after setting values"""

        dac = DAC(n_bits=4)
        for i in range(4):  # set values to expand inputs
            dac.inputs[i] = 0
        self.assertEqual(len(dac.inputs), 4)

        dac = DAC(n_bits=8)
        for i in range(8):  # set values to expand inputs
            dac.inputs[i] = 0
        self.assertEqual(len(dac.inputs), 8)


    def test_sample_zero(self):
        """Test DAC output for zero code"""

        dac = DAC(n_bits=4, span=[-1, 1])

        # Set all bits to 0
        for i in range(4):
            dac.inputs[i] = 0

        # Trigger sampling
        dac.events[0].func_act(0)

        # Code 0 maps to minimum of span
        self.assertEqual(dac.outputs[0], -1.0)


    def test_sample_maximum(self):
        """Test DAC output for maximum code"""

        dac = DAC(n_bits=4, span=[-1, 1])

        # Set all bits to 1 (code = 15)
        for i in range(4):
            dac.inputs[i] = 1

        # Trigger sampling
        dac.events[0].func_act(0)

        # Code 15 maps to maximum of span
        self.assertEqual(dac.outputs[0], 1.0)


    def test_sample_midrange(self):
        """Test DAC output for midrange code"""

        dac = DAC(n_bits=4, span=[-1, 1])

        # Set code to 8 (binary: 1000)
        # LSB=0, bit1=0, bit2=0, MSB=1
        dac.inputs[0] = 0  # LSB
        dac.inputs[1] = 0
        dac.inputs[2] = 0
        dac.inputs[3] = 1  # MSB

        # Trigger sampling
        dac.events[0].func_act(0)

        # Code 8 out of 15 levels
        # scaled_val = 8 / 15 = 0.533...
        # output = -1 + 2 * 0.533... = 0.0666...
        expected = -1.0 + 2.0 * (8.0 / 15.0)
        self.assertAlmostEqual(dac.outputs[0], expected, places=5)


    def test_different_span(self):
        """Test DAC with different output span"""

        dac = DAC(n_bits=2, span=[0, 10])

        # Set code to 2 (binary: 10)
        dac.inputs[0] = 0  # LSB
        dac.inputs[1] = 1  # MSB

        # Trigger sampling
        dac.events[0].func_act(0)

        # Code 2 out of 3 levels (2^2 - 1 = 3)
        # scaled_val = 2 / 3 = 0.666...
        # output = 0 + 10 * 0.666... = 6.666...
        expected = 0.0 + 10.0 * (2.0 / 3.0)
        self.assertAlmostEqual(dac.outputs[0], expected, places=5)


    def test_single_bit_dac(self):
        """Test edge case: 1-bit DAC"""

        dac = DAC(n_bits=1, span=[0, 1])

        # Code 0
        dac.inputs[0] = 0
        dac.events[0].func_act(0)
        self.assertEqual(dac.outputs[0], 0.0)

        # Code 1
        dac.inputs[0] = 1
        dac.events[0].func_act(0)
        self.assertEqual(dac.outputs[0], 1.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)