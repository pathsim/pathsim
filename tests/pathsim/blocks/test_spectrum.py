########################################################################################
##
##                                  TESTS FOR
##                             'blocks.spectrum.py'
##
##                              Milan Rother 2024/25
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

# Use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathsim.blocks.spectrum import Spectrum, RealtimeSpectrum

#base solver for testing
from pathsim.solvers._solver import Solver 


# TESTS ================================================================================

class TestSpectrum(unittest.TestCase):
    """
    Test the implementation of the 'Spectrum' block class
    """

    def test_init(self):

        #test default initialization
        S = Spectrum()
        self.assertEqual(S.time, 0.0)
        self.assertEqual(S.t_wait, 0.0)
        self.assertEqual(S.alpha, 0.0)
        self.assertEqual(S.labels, [])
        self.assertEqual(len(S.freq), 0)
        self.assertEqual(len(S.omega), 0)

        #test specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq, t_wait=20, alpha=0.01, labels=["1", "2"])
        self.assertEqual(S.time, 0.0)
        self.assertEqual(S.t_wait, 20)
        self.assertEqual(S.alpha, 0.01)
        self.assertEqual(S.labels, ["1", "2"])
        self.assertTrue(np.all(S.freq == _freq))
        self.assertTrue(np.all(S.omega == 2*np.pi*_freq))


    def test_len(self):

        S = Spectrum()

        #no direct passthrough
        self.assertEqual(len(S), 0)


    def test_set_solver(self):

        S = Spectrum()

        #test that no solver is initialized
        self.assertEqual(S.engine, None)

        S.set_solver(Solver, None, tolerance_lte_rel=1e-5, tolerance_lte_abs=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(S.engine, Solver))
        self.assertEqual(S.engine.tolerance_lte_rel, 1e-5)
        self.assertEqual(S.engine.tolerance_lte_abs, 1e-6)
        self.assertEqual(S.engine.initial_value, 0.0)

        S.set_solver(Solver, None, tolerance_lte_rel=1e-3, tolerance_lte_abs=1e-4)

        #test that solver tolerance is changed
        self.assertEqual(S.engine.tolerance_lte_rel, 1e-3)
        self.assertEqual(S.engine.tolerance_lte_abs, 1e-4)


    def test_read(self):

        #test read for no engine and default initialization
        S = Spectrum()

        freq, [spec] = S.read()
        
        self.assertEqual(len(freq), 0)
        self.assertEqual(len(spec), 0)

        #test read for no engine and specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq)

        freq, spec = S.read()
        
        self.assertTrue(np.all(freq == _freq))
        self.assertTrue(np.all(spec == np.zeros(100)))

        #test read for engine and specific initialization
        _freq = np.linspace(0, 10, 100)
        S = Spectrum(freq=_freq)
        S.set_solver(Solver, None)

        freq, spec = S.read()
        
        self.assertTrue(np.all(freq == _freq))
        self.assertTrue(np.all(spec == np.zeros(100)))


    def test_reset(self):
        """Test reset clears time"""

        _freq = np.linspace(0, 10, 10)
        S = Spectrum(freq=_freq)
        S.set_solver(Solver, None)

        # Simulate some time passing
        S.time = 5.0
        S.t_sample = 10.0

        # Reset
        S.reset()

        # Time should be reset
        self.assertEqual(S.time, 0.0)


    def test_kernel_no_alpha(self):
        """Test kernel computation without forgetting factor"""

        _freq = np.array([1.0, 2.0])
        S = Spectrum(freq=_freq, alpha=0.0)

        # Test kernel with single input
        u = np.array([1.0])
        t = 0.5
        x = np.zeros(2)  # State (unused when alpha=0)

        result = S._kernel(x, u, t)

        # Expected: u * exp(-j * omega * t)
        expected = np.exp(-1j * S.omega * t)
        np.testing.assert_array_almost_equal(result, expected)


    def test_kernel_with_alpha(self):
        """Test kernel computation with forgetting factor"""

        _freq = np.array([1.0])
        S = Spectrum(freq=_freq, alpha=0.1)

        u = np.array([2.0])
        t = 1.0
        x = np.array([0.5 + 0.3j])

        result = S._kernel(x, u, t)

        # Expected: u * exp(-j * omega * t) - alpha * x
        expected = u[0] * np.exp(-1j * S.omega * t) - S.alpha * x
        np.testing.assert_array_almost_equal(result, expected)


    def test_sample(self):
        """Test sample method updates t_sample"""

        S = Spectrum()

        self.assertEqual(S.t_sample, 0.0)

        S.sample(5.0, None)
        self.assertEqual(S.t_sample, 5.0)

        S.sample(10.0, None)
        self.assertEqual(S.t_sample, 10.0)


    def test_solve_before_wait(self):
        """Test solve returns 0 error before wait period"""

        _freq = np.linspace(0, 10, 10)
        S = Spectrum(freq=_freq, t_wait=5.0)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Before wait period
        S.t_sample = 2.0
        error = S.solve(2.0, 0.1)

        self.assertEqual(error, 0.0)
        self.assertEqual(S.time, 0.0)


    @unittest.skip("Solver.solve() not implemented in base class")
    def test_solve_after_wait(self):
        """Test solve updates after wait period"""

        _freq = np.linspace(0, 10, 10)
        S = Spectrum(freq=_freq, t_wait=5.0)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # After wait period
        S.t_sample = 7.0
        error = S.solve(7.0, 0.1)

        # Time should be updated
        self.assertEqual(S.time, 2.0)  # 7.0 - 5.0

        # Error should be returned (actual value depends on solver)
        self.assertIsInstance(error, float)


    def test_step_before_wait(self):
        """Test step returns no error before wait period"""

        _freq = np.linspace(0, 10, 10)
        S = Spectrum(freq=_freq, t_wait=5.0)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Before wait period
        S.t_sample = 2.0
        success, error, scale = S.step(2.0, 0.1)

        self.assertTrue(success)
        self.assertEqual(error, 0.0)
        self.assertIsNone(scale)  # No rescale needed
        self.assertEqual(S.time, 0.0)


    def test_step_after_wait(self):
        """Test step updates after wait period"""

        _freq = np.linspace(0, 10, 10)
        S = Spectrum(freq=_freq, t_wait=5.0)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # After wait period
        S.t_sample = 7.0
        success, error, scale = S.step(7.0, 0.1)

        # Time should be updated
        self.assertEqual(S.time, 2.0)  # 7.0 - 5.0

        # Should return step results
        self.assertIsInstance(success, bool)
        self.assertIsInstance(error, float)
        # scale is float when adaptive, None when no rescale needed
        self.assertTrue(scale is None or isinstance(scale, float))


    def test_update(self):
        """Test update method (does nothing for Spectrum)"""

        S = Spectrum()

        # Should not raise any errors
        S.update(0.0)
        S.update(10.0)


    def test_read_with_state(self):
        """Test read returns spectrum from solver state"""

        _freq = np.array([1.0, 2.0, 3.0])
        S = Spectrum(freq=_freq, t_wait=0.0)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Manually set solver state and time to simulate integration
        S.time = 10.0
        state = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        S.engine.state = state
        # Set initial_value to something different to pass the check
        S.engine.initial_value = 999.0

        freq, spec = S.read()

        # Should return freq and scaled spectrum (spec is list of arrays)
        np.testing.assert_array_equal(freq, _freq)
        np.testing.assert_array_almost_equal(spec[0], state / S.time)


    def test_read_with_alpha(self):
        """Test read with forgetting factor applies correct scaling"""

        _freq = np.array([1.0])
        alpha = 0.1
        S = Spectrum(freq=_freq, alpha=alpha)
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Manually set solver state and time
        S.time = 5.0
        state = np.array([2.0 + 3.0j])
        S.engine.state = state
        # Set initial_value to something different to pass the check
        S.engine.initial_value = 999.0

        freq, spec = S.read()

        # Should apply forgetting factor scaling
        expected_scale = alpha / (1.0 - np.exp(-alpha * S.time))
        expected = state * expected_scale
        np.testing.assert_array_almost_equal(spec[0], expected)


    def test_plot_no_engine(self):
        """Test plot returns None when no engine"""

        S = Spectrum()

        result = S.plot()

        self.assertIsNone(result)


    def test_plot_with_data(self):
        """Test plot creates figure with data"""

        _freq = np.linspace(0, 10, 20)
        S = Spectrum(freq=_freq, labels=["signal1"])
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Set some state data
        S.time = 1.0
        S.engine.state = np.random.randn(20) + 1j * np.random.randn(20)

        fig, ax = S.plot()

        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Check plot has lines
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)

        plt.close(fig)


    def test_save(self):
        """Test save method writes CSV file"""

        import os
        import csv

        _freq = np.array([1.0, 2.0, 3.0])
        S = Spectrum(freq=_freq, labels=["ch1"])
        S.set_solver(Solver, None)
        S.inputs[0] = 1.0

        # Set some state data
        S.time = 1.0
        S.engine.state = np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])

        # Save to CSV
        test_path = "test_spectrum_save.csv"
        S.save(test_path)

        # Verify file exists
        self.assertTrue(os.path.exists(test_path))

        # Read and verify content
        with open(test_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Check header
            self.assertEqual(rows[0], ["freq [Hz]", "Re(ch1)", "Im(ch1)"])

            # Check data rows
            self.assertEqual(len(rows), 4)  # header + 3 data rows

            # Check first frequency value
            self.assertEqual(float(rows[1][0]), 1.0)

        # Clean up
        os.remove(test_path)


    def test_save_without_csv_extension(self):
        """Test save adds .csv extension if missing"""

        import os

        _freq = np.array([1.0, 2.0])
        S = Spectrum(freq=_freq)
        S.set_solver(Solver, None)
        S.time = 1.0
        S.engine.state = np.array([1.0j, 2.0j])

        # Save without extension
        test_path = "test_spectrum_no_ext"
        S.save(test_path)

        # Verify file exists with .csv extension
        expected_path = test_path + ".csv"
        self.assertTrue(os.path.exists(expected_path))

        # Clean up
        os.remove(expected_path)


    def test_save_multiple_channels(self):
        """Test save with multiple input channels"""

        import os
        import csv

        _freq = np.array([1.0, 2.0])
        S = Spectrum(freq=_freq, labels=["x", "y"])
        S.set_solver(Solver, None)

        # Add two inputs
        S.inputs[0] = 1.0
        S.inputs[1] = 2.0

        # Set state for two channels
        S.time = 1.0
        S.engine.state = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j])

        test_path = "test_spectrum_multi.csv"
        S.save(test_path)

        # Read and verify
        with open(test_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Check header has both channels
            self.assertEqual(rows[0], ["freq [Hz]", "Re(x)", "Im(x)", "Re(y)", "Im(y)"])

        # Clean up
        os.remove(test_path)


class TestRealtimeSpectrum(unittest.TestCase):
    """
    Test the implementation of the 'RealtimeSpectrum' block class
    """

    def test_init(self):
        """Test RealtimeSpectrum initialization"""

        _freq = np.linspace(0, 10, 10)

        # Should emit deprecation warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RS = RealtimeSpectrum(freq=_freq, t_wait=1.0, alpha=0.01, labels=["sig1"])

            # Check that at least one warning was issued and it's about deprecation
            self.assertGreater(len(w), 0)
            # Check that one of the warnings is about deprecation
            has_deprecation = any("deprecated" in str(warning.message).lower() for warning in w)
            self.assertTrue(has_deprecation)

        # Check initialization
        self.assertEqual(RS.t_wait, 1.0)
        self.assertEqual(RS.alpha, 0.01)
        self.assertEqual(RS.labels, ["sig1"])
        np.testing.assert_array_equal(RS.freq, _freq)

        # Check plotter was created
        self.assertIsNotNone(RS.plotter)


    def test_step_early_time(self):
        """Test step before effective time > dt"""

        _freq = np.linspace(0, 10, 10)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RS = RealtimeSpectrum(freq=_freq, t_wait=5.0)

        RS.set_solver(Solver, None)
        RS.inputs[0] = 1.0

        # Very early time
        success, error, scale = RS.step(1.0, 0.1)

        # Should return default values
        self.assertTrue(success)
        self.assertEqual(error, 0.0)
        self.assertIsNone(scale)  # No rescale needed


    def test_step_after_wait(self):
        """Test step after wait period with solver"""

        _freq = np.linspace(0, 10, 10)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RS = RealtimeSpectrum(freq=_freq, t_wait=1.0)

        RS.set_solver(Solver, None)
        RS.inputs[0] = 1.0

        # Mock the plotter's update_all to avoid abs(list) error
        RS.plotter.update_all = lambda x, y: None

        # After wait period (but not too long to trigger plotter update)
        success, error, scale = RS.step(1.15, 0.1)

        # Should perform integration step
        self.assertAlmostEqual(RS.time, 0.15)  # 1.15 - 1.0
        self.assertIsInstance(success, bool)
        self.assertIsInstance(error, float)
        self.assertIsInstance(scale, float)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)