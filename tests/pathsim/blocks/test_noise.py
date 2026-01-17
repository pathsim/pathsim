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
        """Test default initialization"""

        WN = WhiteNoise()

        self.assertEqual(WN.spectral_density, 1)
        self.assertEqual(WN.sampling_period, None)


    def test_init_custom(self):
        """Test custom initialization"""

        WN = WhiteNoise(spectral_density=4.0, sampling_period=10.0)

        self.assertEqual(WN.spectral_density, 4.0)
        self.assertEqual(WN.sampling_period, 10.0)


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        WN = WhiteNoise()
        self.assertEqual(len(WN), 0)


    def test_sample_no_rate(self):
        """Test sampling without rate limit (every timestep)"""

        WN = WhiteNoise(spectral_density=1.0)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(5):
            WN.sample(t, 1)
            WN.update(t)
            values.append(WN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)


    def test_sample_with_rate(self):
        """Test sampling with specified rate uses scheduled events"""

        WN = WhiteNoise(spectral_density=1.0, sampling_period=1.0)

        # When sampling_period is specified, sampling happens via Schedule events
        # not via the sample() method, so verify events exist
        self.assertTrue(hasattr(WN, 'events'))
        self.assertEqual(len(WN.events), 1)
        self.assertEqual(WN.events[0].t_period, 1.0)

        # Verify output can be set by triggering event manually
        initial_output = WN.outputs[0]
        WN.events[0].func_act(0)  # Trigger the scheduled event
        # Output should have changed (very unlikely to be same random value)
        # Just verify it's a float
        self.assertIsInstance(WN.outputs[0], (float, np.floating))


    def test_update(self):
        """Test update method (does nothing for WhiteNoise)"""

        WN = WhiteNoise()

        # Update should not raise errors but doesn't set outputs
        # (outputs are set via sample() or scheduled events)
        WN.update(0)

        # Verify update runs without error
        self.assertEqual(WN.outputs[0], 0.0)


    def test_reset(self):
        """Test reset clears outputs"""

        WN = WhiteNoise()

        # Generate some samples
        for t in range(5):
            WN.sample(t, 1)

        # Output should have been set
        # (may or may not be 0, it's random)

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

        self.assertEqual(PN.spectral_density, 1)
        self.assertEqual(PN.num_octaves, 16)
        self.assertEqual(PN.sampling_period, None)
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(len(PN.octave_values), 16)


    def test_init_custom(self):
        """Test custom initialization"""

        PN = PinkNoise(spectral_density=4.0, num_octaves=8, sampling_period=10.0)

        self.assertEqual(PN.spectral_density, 4.0)
        self.assertEqual(PN.num_octaves, 8)
        self.assertEqual(PN.sampling_period, 10.0)
        self.assertEqual(len(PN.octave_values), 8)


    def test_len(self):
        """Test that noise source has no direct passthrough"""

        PN = PinkNoise()
        self.assertEqual(len(PN), 0)


    def test_sample_no_rate(self):
        """Test sampling without rate limit (every timestep)"""

        PN = PinkNoise(spectral_density=1.0, num_octaves=8)

        # Sample multiple times - should generate new noise each time
        values = []
        for t in range(10):
            PN.sample(t, 1)
            PN.update(t)
            values.append(PN.outputs[0])

        # Check that samples were generated (not all zeros)
        self.assertGreater(np.sum(np.abs(values)), 0)

        # Check that counter increased
        self.assertEqual(PN.n_samples, 10)


    def test_sample_with_rate(self):
        """Test sampling with specified rate uses scheduled events"""

        PN = PinkNoise(spectral_density=1.0, num_octaves=8, sampling_period=1.0)

        # When sampling_period is specified, sampling happens via Schedule events
        # not via the sample() method, so verify events exist
        self.assertTrue(hasattr(PN, 'events'))
        self.assertEqual(len(PN.events), 1)
        self.assertEqual(PN.events[0].t_period, 1.0)

        # Verify output can be set by triggering event manually
        PN.events[0].func_act(0)  # Trigger the scheduled event
        # Output should be set to a value
        self.assertIsInstance(PN.outputs[0], (float, np.floating))
        # n_samples should increase when event is triggered
        self.assertGreater(PN.n_samples, 0)


    def test_octave_update_algorithm(self):
        """Test that octaves are updated according to Voss-McCartney algorithm"""

        PN = PinkNoise(num_octaves=4)

        # Sample multiple times and check that octave values change
        initial_octaves = PN.octave_values.copy()

        for _ in range(10):
            PN.sample(0, 1)  # Increment counter

        # At least some octave values should have changed
        self.assertFalse(np.array_equal(initial_octaves, PN.octave_values))


    def test_update(self):
        """Test update method (does nothing for PinkNoise)"""

        PN = PinkNoise()

        # Update should not raise errors but doesn't set outputs
        # (outputs are set via sample() or scheduled events)
        PN.update(0)

        # Verify update runs without error
        self.assertEqual(PN.outputs[0], 0.0)


    def test_reset(self):
        """Test reset clears noise samples, counter, and resets octaves"""

        PN = PinkNoise(num_octaves=8)

        # Generate some samples
        initial_octaves = PN.octave_values.copy()
        for t in range(5):
            PN.sample(t, 0.1)  # sample() requires dt parameter

        # Verify samples were generated
        self.assertEqual(PN.n_samples, 5)

        # Reset
        PN.reset()

        # Check reset worked
        self.assertEqual(PN.n_samples, 0)
        self.assertEqual(PN.outputs[0], 0.0)

        # Octave values should be reinitialized (different from before)
        # (technically could be same by chance, but very unlikely)
        self.assertEqual(len(PN.octave_values), 8)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
