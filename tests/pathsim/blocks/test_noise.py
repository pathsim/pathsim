########################################################################################
##
##                                  TESTS FOR
##                               'blocks.noise.py'
##
##                              Milan Rother 2024/25
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.noise import WhiteNoise, PinkNoise


# TESTS ================================================================================

class TestWhiteNoise(unittest.TestCase):
    """
    Test the implementation of the 'WhiteNoise' block class
    """

    def test_init_default(self):
        """Test default initialization uses standard_deviation mode"""

        WN = WhiteNoise()

        self.assertEqual(WN.standard_deviation, 1.0)
        self.assertIsNone(WN.spectral_density)
        self.assertIsNone(WN.sampling_period)


    def test_init_standard_deviation(self):
        """Test initialization with standard_deviation parameter"""

        WN = WhiteNoise(standard_deviation=2.5)

        self.assertEqual(WN.standard_deviation, 2.5)
        self.assertIsNone(WN.spectral_density)


    def test_init_spectral_density(self):
        """Test initialization with spectral_density parameter"""

        WN = WhiteNoise(spectral_density=4.0, sampling_period=0.1)

        self.assertEqual(WN.spectral_density, 4.0)
        self.assertEqual(WN.sampling_period, 0.1)


    def test_init_with_seed(self):
        """Test that seed parameter produces reproducible results"""

        WN1 = WhiteNoise(standard_deviation=1.0, seed=42)
        WN2 = WhiteNoise(standard_deviation=1.0, seed=42)

        # Generate samples from both
        WN1.sample(0, 0.01)
        WN2.sample(0, 0.01)

        self.assertEqual(WN1.outputs[0], WN2.outputs[0])


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        WN = WhiteNoise()
        self.assertEqual(len(WN), 0)


    def test_sample_continuous_mode(self):
        """Test sampling in continuous mode (every timestep)"""

        WN = WhiteNoise(standard_deviation=1.0)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(5):
            WN.sample(t, 0.01)
            values.append(WN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)

        # Check that values are different (very unlikely to be same)
        self.assertFalse(all(v == values[0] for v in values))


    def test_sample_discrete_mode(self):
        """Test sampling with discrete sampling_period"""

        WN = WhiteNoise(standard_deviation=1.0, sampling_period=0.1)

        # When sampling_period is specified, events should be created
        self.assertTrue(hasattr(WN, 'events'))
        self.assertEqual(len(WN.events), 1)
        self.assertEqual(WN.events[0].t_period, 0.1)

        # Initial output should be set (not zero)
        self.assertIsInstance(WN.outputs[0], (float, np.floating))

        # Triggering event should update output
        old_output = WN.outputs[0]
        WN.events[0].func_act(0)
        # Very unlikely to get same random value
        self.assertIsInstance(WN.outputs[0], (float, np.floating))


    def test_spectral_density_scaling(self):
        """Test that spectral_density mode scales with timestep"""

        WN = WhiteNoise(spectral_density=1.0, seed=123)

        # With spectral density mode, variance should scale as S0/dt
        # Smaller dt -> larger amplitude
        WN.sample(0, 0.01)  # dt = 0.01
        val_small_dt = WN.outputs[0]

        WN2 = WhiteNoise(spectral_density=1.0, seed=123)
        WN2.sample(0, 1.0)  # dt = 1.0
        val_large_dt = WN2.outputs[0]

        # Same seed means same underlying N(0,1) sample
        # So ratio should be sqrt(1.0/0.01) = 10
        self.assertAlmostEqual(abs(val_small_dt / val_large_dt), 10.0, places=5)


    def test_standard_deviation_constant_amplitude(self):
        """Test that standard_deviation mode gives constant amplitude"""

        WN = WhiteNoise(standard_deviation=2.0, seed=456)

        # Generate many samples and check standard deviation
        samples = []
        for i in range(1000):
            WN.sample(i, 0.01)
            samples.append(WN.outputs[0])

        # Standard deviation should be close to 2.0
        self.assertAlmostEqual(np.std(samples), 2.0, delta=0.2)


    def test_update(self):
        """Test update method doesn't change output"""

        WN = WhiteNoise()
        WN.sample(0, 0.01)
        old_output = WN.outputs[0]
        WN.update(0)
        self.assertEqual(WN.outputs[0], old_output)


    def test_reset(self):
        """Test reset clears outputs"""

        WN = WhiteNoise()

        # Generate some samples
        for t in range(5):
            WN.sample(t, 0.01)

        # Reset
        WN.reset()

        # Check reset worked - output should be back to 0
        self.assertEqual(WN.outputs[0], 0.0)


class TestPinkNoise(unittest.TestCase):
    """
    Test the implementation of the 'PinkNoise' block class
    """

    def test_init_default(self):
        """Test default initialization"""

        PN = PinkNoise()

        self.assertEqual(PN.standard_deviation, 1.0)
        self.assertIsNone(PN.spectral_density)
        self.assertEqual(PN.num_octaves, 16)
        self.assertIsNone(PN.sampling_period)
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(len(PN.octave_values), 16)


    def test_init_standard_deviation(self):
        """Test initialization with standard_deviation"""

        PN = PinkNoise(standard_deviation=0.5, num_octaves=8)

        self.assertEqual(PN.standard_deviation, 0.5)
        self.assertIsNone(PN.spectral_density)
        self.assertEqual(PN.num_octaves, 8)
        self.assertEqual(len(PN.octave_values), 8)


    def test_init_spectral_density(self):
        """Test initialization with spectral_density"""

        PN = PinkNoise(spectral_density=4.0, num_octaves=8, sampling_period=0.1)

        self.assertEqual(PN.spectral_density, 4.0)
        self.assertEqual(PN.num_octaves, 8)
        self.assertEqual(PN.sampling_period, 0.1)


    def test_init_with_seed(self):
        """Test that seed parameter produces reproducible results"""

        PN1 = PinkNoise(standard_deviation=1.0, num_octaves=8, seed=42)
        PN2 = PinkNoise(standard_deviation=1.0, num_octaves=8, seed=42)

        # Octave values should be identical
        self.assertTrue(np.array_equal(PN1.octave_values, PN2.octave_values))

        # Generate samples from both
        PN1.sample(0, 0.01)
        PN2.sample(0, 0.01)

        self.assertEqual(PN1.outputs[0], PN2.outputs[0])


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        PN = PinkNoise()
        self.assertEqual(len(PN), 0)


    def test_sample_continuous_mode(self):
        """Test sampling in continuous mode (every timestep)"""

        PN = PinkNoise(standard_deviation=1.0, num_octaves=8)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(10):
            PN.sample(t, 0.01)
            values.append(PN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)

        # Check that counter increased
        self.assertEqual(PN.n_samples, 10)


    def test_sample_discrete_mode(self):
        """Test sampling with discrete sampling_period"""

        PN = PinkNoise(standard_deviation=1.0, num_octaves=8, sampling_period=0.1)

        # When sampling_period is specified, events should be created
        self.assertTrue(hasattr(PN, 'events'))
        self.assertEqual(len(PN.events), 1)
        self.assertEqual(PN.events[0].t_period, 0.1)

        # Initial output should be set
        self.assertIsInstance(PN.outputs[0], (float, np.floating))

        # Triggering event should update output and counter
        PN.events[0].func_act(0)
        self.assertIsInstance(PN.outputs[0], (float, np.floating))
        self.assertGreater(PN.n_samples, 0)


    def test_octave_update_algorithm(self):
        """Test that octaves are updated according to Voss-McCartney algorithm"""

        PN = PinkNoise(num_octaves=4, seed=123)

        # Sample multiple times and check that octave values change
        initial_octaves = PN.octave_values.copy()

        for _ in range(10):
            PN.sample(0, 0.01)

        # At least some octave values should have changed
        self.assertFalse(np.array_equal(initial_octaves, PN.octave_values))


    def test_update(self):
        """Test update method doesn't change output"""

        PN = PinkNoise()
        PN.sample(0, 0.01)
        old_output = PN.outputs[0]
        PN.update(0)
        self.assertEqual(PN.outputs[0], old_output)


    def test_reset(self):
        """Test reset clears noise samples, counter, and resets octaves"""

        PN = PinkNoise(num_octaves=8, seed=789)

        # Generate some samples
        for t in range(5):
            PN.sample(t, 0.01)

        # Verify samples were generated
        self.assertEqual(PN.n_samples, 5)

        # Reset
        PN.reset()

        # Check reset worked
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(PN.outputs[0], 0.0)
        self.assertEqual(PN._current_sample, 0.0)

        # Octave values should be reinitialized
        self.assertEqual(len(PN.octave_values), 8)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
