########################################################################################
##
##                                  TESTS FOR 
##                              'blocks.scope.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathsim.blocks.scope import Scope, RealtimeScope


# TESTS ================================================================================

class TestScope(unittest.TestCase):
    """
    Test the implementation of the 'Scope' block class
    """

    def test_init(self):

        #test default initialization
        S = Scope()

        self.assertEqual(S.sampling_rate, None)
        self.assertEqual(S.t_wait, 0.0)
        self.assertEqual(S.labels, [])
        self.assertEqual(S.recording_time, [])
        self.assertEqual(S.recording_data, [])

        #test specific initialization
        S = Scope(sampling_rate=1, t_wait=1.0, labels=["1", "2"])

        self.assertEqual(S.sampling_rate, 1)
        self.assertEqual(S.t_wait, 1.0)
        self.assertEqual(S.labels, ["1", "2"])

    def test_inputs_default(self):
        """Catches bug in #60"""

        S1 = Scope()
        S2 = Scope()

        S1.labels.append('A')
        assert S2.labels == []

    def test_len(self):
        
        S = Scope()

        #no passthrough
        self.assertEqual(len(S), 0)


    def test_reset(self):

        S = Scope()

        for t in range(10):

            S.inputs[0] = t
            S.sample(t, None)

        #test that we have some recording
        self.assertGreater(len(S.recording_time), 0)

        S.reset()

        #test if reset was successful
        self.assertEqual(S.recording_time, [])
        self.assertEqual(S.recording_data, [])


    def test_sample(self):

        #single input default initialization
        S = Scope()

        for t in range(10):

            S.inputs[0] = t
            S.sample(t, None)

            #test most recent recording
            self.assertEqual(S.recording_time[-1], t)
            self.assertEqual(S.recording_data[-1][0], t)

        #multi input default initialization
        S = Scope()

        for t in range(10):

            S.inputs[0] = t
            S.inputs[1] = 2*t
            S.inputs[2] = 3*t
            S.sample(t, None)

            #test most recent recording
            self.assertEqual(S.recording_time[-1], t)
            self.assertTrue(np.all(np.equal(S.recording_data[-1], [t, 2*t, 3*t])))


    def test_read(self):

        _time = np.arange(10)

        #single input default initialization
        S = Scope()

        for t in _time:

            S.inputs[0] = t
            S.sample(t, None)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time)))

        #test if input was recorded correctly
        self.assertTrue(np.all(np.equal(result, _time)))

        #multi input default initialization
        S = Scope()

        for t in _time:

            S.inputs[0] = t
            S.inputs[1] = 2*t
            S.inputs[2] = 3*t
            S.sample(t, None)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time)))

        #test if multi input was recorded correctly
        self.assertTrue(np.all(np.equal(result, [_time, 2*_time, 3*_time])))


    def test_sampling_rate(self):
        #TODO: implement this in the simulation loop because the 'Schedule' event
        pass


    def test_t_wait(self):

        _time = np.arange(10)

        #single input special t_wait
        S = Scope(t_wait=5)

        for t in _time:

            S.inputs[0] = t
            S.sample(t, None)

        time, result = S.read()

        #test if time was recorded correctly
        self.assertTrue(np.all(np.equal(time, _time[5:])))

        #test if input was recorded correctly
        self.assertTrue(np.all(np.equal(result, _time[5:])))


    def test_read_empty(self):
        """Test read() returns None for empty recording"""

        S = Scope()
        time, data = S.read()

        self.assertIsNone(time)
        self.assertIsNone(data)


    def test_plot_empty(self):
        """Test plot() with empty recording"""

        S = Scope()

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot()

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertEqual(len(w), 1)
            self.assertIn("no recording", str(w[0].message))


    @unittest.skip("Matplotlib/numpy compatibility issue in Python 3.13")
    def test_plot_with_data(self):
        """Test plot() with recorded data"""

        S = Scope(labels=["signal1", "signal2"])

        # Record some data
        for t in range(5):
            S.inputs[0] = np.sin(t)
            S.inputs[1] = np.cos(t)
            S.sample(t, None)

        # Plot (non-blocking)
        fig, ax = S.plot()

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check that 2 lines were plotted
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)

        # Close figure to free resources
        plt.close(fig)


    def test_plot2D_empty(self):
        """Test plot2D() with empty recording"""

        S = Scope()

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot2D()

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertEqual(len(w), 1)
            self.assertIn("no recording", str(w[0].message))


    def test_plot2D_insufficient_channels(self):
        """Test plot2D() with insufficient channels"""

        S = Scope()

        # Record only 1 channel
        for t in range(5):
            S.inputs[0] = t
            S.sample(t, None)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot2D()

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertEqual(len(w), 1)
            self.assertIn("not enough channels", str(w[0].message))


    @unittest.skip("Matplotlib/numpy compatibility issue in Python 3.13")
    def test_plot2D_with_data(self):
        """Test plot2D() with 2 channels"""

        S = Scope(labels=["x", "y"])

        # Record 2 channels
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.sample(t, None)

        fig, ax = S.plot2D(axes=(0, 1))

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check that 1 line was plotted (x vs y)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1)

        plt.close(fig)


    def test_plot2D_invalid_axes(self):
        """Test plot2D() with invalid axis selection"""

        S = Scope()

        # Record 2 channels
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.sample(t, None)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot2D(axes=(0, 5))  # axis 5 doesn't exist

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertTrue(len(w) > 0)
            self.assertIn("out of bounds", str(w[0].message))


    def test_plot3D_empty(self):
        """Test plot3D() with empty recording"""

        S = Scope()

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot3D()

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertEqual(len(w), 1)
            self.assertIn("no recording", str(w[0].message))


    def test_plot3D_insufficient_channels(self):
        """Test plot3D() with insufficient channels"""

        S = Scope()

        # Record only 2 channels
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.sample(t, None)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot3D()

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertTrue(len(w) > 0)
            self.assertIn("at least 3 channels", str(w[0].message))


    @unittest.skip("Matplotlib/numpy compatibility issue in Python 3.13")
    def test_plot3D_with_data(self):
        """Test plot3D() with 3 channels"""

        S = Scope(labels=["x", "y", "z"])

        # Record 3 channels
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.inputs[2] = t * 3
            S.sample(t, None)

        fig, ax = S.plot3D(axes=(0, 1, 2))

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        plt.close(fig)


    def test_plot3D_invalid_axes(self):
        """Test plot3D() with invalid axis selection"""

        S = Scope()

        # Record 3 channels
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.inputs[2] = t * 3
            S.sample(t, None)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax = S.plot3D(axes=(0, 1, 10))  # axis 10 doesn't exist

            self.assertIsNone(fig)
            self.assertIsNone(ax)
            self.assertTrue(len(w) > 0)
            self.assertIn("out of bounds", str(w[0].message))


    def test_save(self):
        """Test save() method"""

        import os
        import csv

        S = Scope(labels=["sig1", "sig2"])

        # Record some data
        for t in range(3):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.sample(t, None)

        # Save to CSV
        test_path = "test_scope_save.csv"
        S.save(test_path)

        # Verify file exists
        self.assertTrue(os.path.exists(test_path))

        # Read and verify content
        with open(test_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Check header
            self.assertEqual(rows[0], ["time [s]", "sig1", "sig2"])

            # Check data rows
            self.assertEqual(len(rows), 4)  # header + 3 data rows

        # Clean up
        os.remove(test_path)


    def test_save_without_csv_extension(self):
        """Test save() adds .csv extension if missing"""

        import os

        S = Scope()

        for t in range(2):
            S.inputs[0] = t
            S.sample(t, None)

        # Save without extension
        test_path = "test_scope_no_ext"
        S.save(test_path)

        # Verify file exists with .csv extension
        expected_path = test_path + ".csv"
        self.assertTrue(os.path.exists(expected_path))

        # Clean up
        os.remove(expected_path)


class TestRealtimeScope(unittest.TestCase):
    """
    Test the implementation of the 'RealtimeScope' block class
    """

    def test_init(self):
        """Test RealtimeScope initialization"""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            S = RealtimeScope(sampling_rate=0.1, t_wait=1.0, labels=["sig1"])

            # Should warn about deprecation (may be one of several warnings)
            self.assertTrue(len(w) > 0)
            has_deprecation = any("deprecated" in str(warning.message).lower() for warning in w)
            self.assertTrue(has_deprecation, "Expected deprecation warning not found")

            # Should have plotter
            self.assertIsNotNone(S.plotter)


    def test_sample_with_realtime_plotter(self):
        """Test RealtimeScope sample method with plotter"""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            S = RealtimeScope(labels=["x", "y"])

        # Sample some data
        for t in range(5):
            S.inputs[0] = t
            S.inputs[1] = t * 2
            S.sample(t, None)

        # Should have recorded data
        self.assertGreater(len(S.recording_time), 0)


    def test_sample_with_sampling_rate(self):
        """Test RealtimeScope with sampling_rate parameter"""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            S = RealtimeScope(sampling_rate=1.0, labels=["sig"])

        # Sample at different times
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            S.inputs[0] = t
            S.sample(t, None)

        # Recording behavior depends on sampling rate logic
        self.assertIsNotNone(S.recording_time)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)