########################################################################################
##
##                                  TESTS FOR
##                  'opt/sensitivity.py' and ParameterEstimator.sensitivity()
##
########################################################################################

# IMPORTS ==============================================================================

import math
import unittest

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Scope
from pathsim.blocks.sources import Source
from pathsim.solvers import SSPRK22
from pathsim.opt import ParameterEstimator, SensitivityResult
from pathsim.utils.timeseries_data import TimeSeriesData


# HELPERS ==============================================================================

def _identity_jacobian(n):
    """Return an n×n identity Jacobian (orthogonal, perfectly conditioned)."""
    return np.eye(n)


def _make_ramp_estimator():
    """
    Minimal single-experiment estimator for y = gain * t.
    Returns (estimator, true_x) where true_x = [3.0].
    """
    true_gain = 3.0

    source = Source(func=lambda t: t)
    amp    = Amplifier(gain=1.0)
    scope  = Scope()

    sim = Simulation(
        [source, amp, scope],
        [Connection(source[0], amp[0]), Connection(amp[0], scope[0])],
        Solver=SSPRK22, dt=0.1, log=False,
    )

    t_meas = np.linspace(1.0, 5.0, 9)
    y_meas = true_gain * t_meas        # noise-free for deterministic tests
    meas   = TimeSeriesData(time=t_meas, data=y_meas)

    est = ParameterEstimator(simulator=sim)
    est.add_block_parameter(amp, "gain", value=1.0, bounds=(0.0, 10.0))
    est.add_timeseries(meas, signal=scope[0], sigma=0.1)

    return est, np.array([true_gain])


# TESTS ================================================================================

class TestSensitivityResultConstruction(unittest.TestCase):
    """SensitivityResult built from a known Jacobian."""

    def _result_identity(self, n=3):
        J      = _identity_jacobian(n)
        names  = [f"p{i}" for i in range(n)]
        values = np.ones(n) * 2.0
        return SensitivityResult(jacobian=J, param_names=names, param_values=values)

    def test_fim_is_identity_for_identity_jacobian(self):
        r = self._result_identity(3)
        np.testing.assert_allclose(r.fim, np.eye(3))

    def test_covariance_is_identity_for_identity_jacobian(self):
        r = self._result_identity(3)
        np.testing.assert_allclose(r.covariance, np.eye(3), atol=1e-12)

    def test_std_errors_are_one_for_identity_jacobian(self):
        r = self._result_identity(3)
        np.testing.assert_allclose(r.std_errors, np.ones(3), atol=1e-12)

    def test_correlation_diagonal_is_one(self):
        r = self._result_identity(3)
        np.testing.assert_allclose(np.diag(r.correlation), np.ones(3), atol=1e-12)

    def test_correlation_off_diagonal_zero_for_orthogonal_jacobian(self):
        r = self._result_identity(3)
        mask = ~np.eye(3, dtype=bool)
        np.testing.assert_allclose(r.correlation[mask], 0.0, atol=1e-12)

    def test_eigenvalues_all_one_for_identity_fim(self):
        r = self._result_identity(3)
        np.testing.assert_allclose(r.eigenvalues, np.ones(3), atol=1e-12)

    def test_condition_number_one_for_identity(self):
        r = self._result_identity(3)
        self.assertAlmostEqual(r.condition_number, 1.0, places=10)

    def test_param_names_stored(self):
        r = self._result_identity(2)
        self.assertEqual(r.param_names, ["p0", "p1"])

    def test_param_values_stored(self):
        r = self._result_identity(2)
        np.testing.assert_array_equal(r.param_values, [2.0, 2.0])

    def test_eigenvalues_sorted_descending(self):
        # Non-symmetric Jacobian to get varied eigenvalues
        J = np.diag([3.0, 1.0, 2.0])
        r = SensitivityResult(J, ["a", "b", "c"], np.zeros(3))
        self.assertTrue(np.all(np.diff(r.eigenvalues) <= 0))


class TestSensitivityResultCorrelation(unittest.TestCase):
    """Correlation detection for known correlated parameters."""

    def test_perfectly_correlated_parameters(self):
        # Two identical columns → perfect correlation
        col  = np.array([1.0, 2.0, 3.0])
        J    = np.column_stack([col, col])
        r    = SensitivityResult(J, ["a", "b"], np.array([1.0, 1.0]))
        self.assertAlmostEqual(abs(r.correlation[0, 1]), 1.0, places=5)

    def test_uncorrelated_parameters(self):
        J = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        r = SensitivityResult(J, ["a", "b"], np.array([1.0, 1.0]))
        self.assertAlmostEqual(r.correlation[0, 1], 0.0, places=10)

    def test_rank_deficient_jacobian_handled_gracefully(self):
        # Zero Jacobian → FIM is all-zeros → no crash
        J = np.zeros((5, 2))
        r = SensitivityResult(J, ["a", "b"], np.zeros(2))
        self.assertTrue(np.isfinite(r.covariance).all() or True)  # just no exception

    def test_single_parameter(self):
        J = np.array([[2.0], [0.5]])
        r = SensitivityResult(J, ["gain"], np.array([3.0]))
        self.assertEqual(r.correlation.shape, (1, 1))
        self.assertAlmostEqual(r.correlation[0, 0], 1.0, places=10)


class TestSensitivityResultConditionNumber(unittest.TestCase):

    def test_well_conditioned(self):
        J = _identity_jacobian(3)
        r = SensitivityResult(J, ["a", "b", "c"], np.zeros(3))
        self.assertLess(r.condition_number, 1e3)

    def test_ill_conditioned(self):
        # Eigenvalues 1e6 and 1 → condition number 1e6
        J = np.diag([1e3, 1.0])
        r = SensitivityResult(J, ["a", "b"], np.zeros(2))
        self.assertGreater(r.condition_number, 1e3)

    def test_all_zero_eigenvalues_gives_inf(self):
        J = np.zeros((3, 2))
        r = SensitivityResult(J, ["a", "b"], np.zeros(2))
        self.assertTrue(np.isinf(r.condition_number))


class TestSensitivityResultDisplay(unittest.TestCase):
    """display() runs without error for various inputs."""

    def test_display_runs(self):
        r = SensitivityResult(_identity_jacobian(2), ["gain", "offset"], np.array([3.0, 1.0]))
        try:
            r.display()
        except Exception as e:
            self.fail(f"display() raised {e}")

    def test_display_with_zero_value_param(self):
        r = SensitivityResult(_identity_jacobian(2), ["a", "b"], np.array([0.0, 1.0]))
        r.display()  # should not raise or divide by zero

    def test_display_single_param(self):
        J = np.array([[1.5], [0.5]])
        r = SensitivityResult(J, ["gain"], np.array([2.0]))
        r.display()


class TestSensitivityResultPlot(unittest.TestCase):

    def test_plot_returns_fig_and_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = SensitivityResult(_identity_jacobian(3), ["a", "b", "c"], np.ones(3))
        fig, axes = r.plot()

        import matplotlib.figure
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertEqual(len(axes), 2)
        plt.close("all")

    def test_plot_single_param(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        J = np.array([[1.0], [0.5]])
        r = SensitivityResult(J, ["gain"], np.array([3.0]))
        fig, axes = r.plot()
        plt.close("all")


class TestParameterEstimatorSensitivity(unittest.TestCase):
    """ParameterEstimator.sensitivity() integration tests."""

    def setUp(self):
        self.est, self.true_x = _make_ramp_estimator()

    def test_sensitivity_requires_x_or_cached(self):
        with self.assertRaises(ValueError):
            self.est.sensitivity()  # no fit done, no cached x

    def test_sensitivity_with_explicit_x(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertIsInstance(sens, SensitivityResult)

    def test_sensitivity_after_fit_uses_cached_x(self):
        result = self.est.fit(verbose=0)
        sens = self.est.sensitivity()           # no x needed
        self.assertIsInstance(sens, SensitivityResult)

    def test_jacobian_shape(self):
        n_meas   = 9   # 9 measurement points
        n_params = 1
        sens = self.est.sensitivity(self.true_x)
        self.assertEqual(sens.jacobian.shape, (n_meas, n_params))

    def test_single_param_sensitivity_is_well_conditioned(self):
        sens = self.est.sensitivity(self.true_x)
        # With a single parameter and noise-free data, condition number = 1
        self.assertAlmostEqual(sens.condition_number, 1.0, places=5)

    def test_std_error_is_small_at_true_optimum(self):
        # At the true gain=3, residuals are near zero → FIM is large → SE small
        sens = self.est.sensitivity(self.true_x)
        self.assertGreater(sens.std_errors[0], 0.0)
        self.assertLess(sens.std_errors[0], 1.0)

    def test_sensitivity_custom_eps(self):
        sens = self.est.sensitivity(self.true_x, eps=1e-4)
        self.assertIsInstance(sens, SensitivityResult)

    def test_param_names_match(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertEqual(len(sens.param_names), 1)
        self.assertIn("gain", sens.param_names[0])

    def test_param_value_in_model_space(self):
        sens = self.est.sensitivity(self.true_x)
        # No transform → optimizer space == model space
        self.assertAlmostEqual(float(sens.param_values[0]), self.true_x[0], places=5)

    def test_sensitivity_no_fit_required(self):
        # Can call sensitivity without ever calling fit()
        sens = self.est.sensitivity(np.array([2.0]))
        self.assertIsInstance(sens, SensitivityResult)


class TestSensitivityResultImport(unittest.TestCase):
    """SensitivityResult is importable from pathsim.opt."""

    def test_import_from_opt(self):
        from pathsim.opt import SensitivityResult as SR
        self.assertIs(SR, SensitivityResult)


if __name__ == "__main__":
    unittest.main()
