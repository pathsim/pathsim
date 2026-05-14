########################################################################################
##
##                                  TESTS FOR
##           SchurResult  and  ParameterEstimator.sensitivity()  (Schur path)
##
########################################################################################

# IMPORTS ==============================================================================

import unittest

import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Amplifier, Scope
from pathsim.blocks.sources import Constant, Source
from pathsim.blocks.adder import Adder
from pathsim.solvers import SSPRK22
from pathsim.opt import ParameterEstimator, SchurResult, SensitivityResult
from pathsim.utils.timeseries_data import TimeSeriesData


# HELPERS ==============================================================================

def _make_additive_multi_exp_estimator(n_exp=2):
    """
    Multi-experiment estimator for y(t) = global_gain * t + local_offset_i.

    Global: Amplifier.gain  (slope, shared across all experiments)
    Local:  Constant.value  (intercept, per experiment)

    True values:
      global_gain = 2.0
      local_offset_0 = 1.0, local_offset_1 = 5.0, ...

    The Jacobian columns are:
      col 0   (global): ∂y/∂gain   = t          (varies with t)
      col 1+i (local):  ∂y/∂offset = 1           (constant)

    These are linearly independent, so S_G is well-conditioned and finite.
    """
    true_global = 2.0
    true_locals = [1.0 + 4.0 * i for i in range(n_exp)]   # [1, 5, 9, …]

    source      = Source(func=lambda t: t)
    global_amp  = Amplifier(gain=true_global)
    local_const = Constant(value=0.0)     # local; will be varied per experiment
    adder       = Adder()
    scope       = Scope()

    sim = Simulation(
        [source, global_amp, local_const, adder, scope],
        [
            Connection(source[0],     global_amp[0]),
            Connection(global_amp[0], adder[0]),
            Connection(local_const[0], adder[1]),
            Connection(adder[0],      scope[0]),
        ],
        Solver=SSPRK22, dt=0.1, log=False,
    )

    t_meas = np.linspace(0.5, 4.0, 8)

    # First experiment reuses the original sim; subsequent ones are deep copies.
    est = ParameterEstimator(simulator=sim, adaptive=False)
    for _ in range(1, n_exp):
        est.add_experiment(sim, copy_sim=True)

    for i in range(n_exp):
        y_meas = true_global * t_meas + true_locals[i]
        meas   = TimeSeriesData(time=t_meas, data=y_meas)
        est.add_timeseries(meas, signal=scope[0], experiment=i)

    # Register global then local parameters
    est.add_global_block_parameter(
        "Amplifier", "gain", value=1.0, bounds=(-10.0, 10.0)
    )
    for i in range(n_exp):
        est.add_local_block_parameter(
            i, "Constant", "value", value=0.0, bounds=(-20.0, 20.0)
        )

    true_x = np.array([true_global] + true_locals)
    return est, true_x


# TESTS ================================================================================

class TestSchurResultConstruction(unittest.TestCase):
    """SchurResult built from a known Schur FIM."""

    def _identity_schur(self, n=2):
        fim = np.eye(n)
        names  = [f"g{i}" for i in range(n)]
        values = np.ones(n) * 3.0
        return SchurResult(schur_fim=fim, param_names=names, param_values=values)

    def test_fim_stored(self):
        r = self._identity_schur(2)
        np.testing.assert_allclose(r.fim, np.eye(2))

    def test_covariance_is_identity_for_identity_fim(self):
        r = self._identity_schur(3)
        np.testing.assert_allclose(r.covariance, np.eye(3), atol=1e-12)

    def test_std_errors_are_one_for_identity_fim(self):
        r = self._identity_schur(3)
        np.testing.assert_allclose(r.std_errors, np.ones(3), atol=1e-12)

    def test_correlation_diagonal_is_one(self):
        r = self._identity_schur(2)
        np.testing.assert_allclose(np.diag(r.correlation), np.ones(2), atol=1e-12)

    def test_condition_number_one_for_identity(self):
        r = self._identity_schur(3)
        self.assertAlmostEqual(r.condition_number, 1.0, places=10)

    def test_ill_conditioned(self):
        fim = np.diag([1e6, 1.0])
        r = SchurResult(fim, ["a", "b"], np.zeros(2))
        self.assertGreater(r.condition_number, 1e3)

    def test_rank_deficient_gives_inf(self):
        fim = np.zeros((2, 2))
        r = SchurResult(fim, ["a", "b"], np.zeros(2))
        self.assertTrue(np.isinf(r.condition_number))

    def test_single_parameter(self):
        fim = np.array([[4.0]])
        r = SchurResult(fim, ["k"], np.array([1.0]))
        self.assertAlmostEqual(r.condition_number, 1.0, places=10)
        self.assertAlmostEqual(r.std_errors[0], 0.5, places=10)

    def test_eigenvalues_descending(self):
        fim = np.diag([3.0, 1.0, 2.0])
        r = SchurResult(fim, ["a", "b", "c"], np.zeros(3))
        self.assertTrue(np.all(np.diff(r.eigenvalues) <= 0))

    def test_param_names_stored(self):
        r = self._identity_schur(2)
        self.assertEqual(r.param_names, ["g0", "g1"])

    def test_param_values_stored(self):
        r = self._identity_schur(2)
        np.testing.assert_array_equal(r.param_values, [3.0, 3.0])


class TestSchurResultDisplay(unittest.TestCase):

    def test_display_runs(self):
        r = SchurResult(np.eye(2), ["k", "V"], np.array([0.5, 10.0]))
        try:
            r.display()
        except Exception as e:
            self.fail(f"display() raised {e}")

    def test_display_with_zero_value(self):
        r = SchurResult(np.eye(2), ["a", "b"], np.array([0.0, 1.0]))
        r.display()  # must not raise

    def test_display_single_param(self):
        r = SchurResult(np.array([[2.0]]), ["k"], np.array([0.5]))
        r.display()


class TestSchurResultPlot(unittest.TestCase):

    def test_plot_returns_fig_and_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = SchurResult(np.eye(2), ["a", "b"], np.ones(2))
        fig, axes = r.plot()

        import matplotlib.figure
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        self.assertEqual(len(axes), 2)
        plt.close("all")

    def test_plot_single_param(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = SchurResult(np.array([[4.0]]), ["k"], np.array([0.5]))
        fig, axes = r.plot()
        plt.close("all")


class TestSchurMath(unittest.TestCase):
    """Verify the Schur complement arithmetic on synthetic Jacobians."""

    def test_schur_reduces_to_F_GG_when_locals_are_orthogonal(self):
        """When cross term F_GLi = 0, S_G == F_GG."""
        # J_G = [1, 1, 1, 1]^T, J_L = [1, -1, 0, 0]^T  (J_G^T J_L = 0)
        J_G = np.array([[1.0], [1.0], [1.0], [1.0]])
        J_L = np.array([[1.0], [-1.0], [0.0], [0.0]])
        F_GG = J_G.T @ J_G
        F_LL = J_L.T @ J_L
        F_GL = J_G.T @ J_L            # [[0]] → no Schur correction
        S_G  = F_GG - F_GL @ np.linalg.pinv(F_LL) @ F_GL.T
        np.testing.assert_allclose(S_G, F_GG, atol=1e-12)

    def test_schur_less_than_F_GG_when_correlated(self):
        """When J_G ∝ J_L, S_G ≈ 0 (local absorbs all global information)."""
        J_G  = np.array([[1.0], [2.0], [3.0]])
        J_L  = J_G.copy()
        F_GG = J_G.T @ J_G
        F_LL = J_L.T @ J_L
        F_GL = J_G.T @ J_L
        S_G  = F_GG - F_GL @ np.linalg.pinv(F_LL) @ F_GL.T
        self.assertLessEqual(float(S_G[0, 0]), float(F_GG[0, 0]))
        self.assertAlmostEqual(float(S_G[0, 0]), 0.0, places=10)

    def test_two_uncorrelated_experiments_no_correction(self):
        """Two experiments with orthogonal locals: S_G == F_GG."""
        J_G0 = np.array([[1.0], [1.0]])
        J_L0 = np.array([[1.0], [-1.0]])   # F_GL0 = 0
        J_G1 = np.array([[1.0], [1.0]])
        J_L1 = np.array([[1.0], [-1.0]])   # F_GL1 = 0

        F_GG = J_G0.T @ J_G0 + J_G1.T @ J_G1
        S_G  = F_GG.copy()
        for J_G, J_L in [(J_G0, J_L0), (J_G1, J_L1)]:
            F_LL = J_L.T @ J_L
            F_GL = J_G.T @ J_L
            S_G -= F_GL @ np.linalg.pinv(F_LL) @ F_GL.T

        np.testing.assert_allclose(S_G, F_GG, atol=1e-12)

    def test_schur_is_non_negative_definite(self):
        """S_G must always be positive semi-definite."""
        rng = np.random.default_rng(7)
        J_G = rng.standard_normal((10, 2))
        J_L = rng.standard_normal((10, 3))
        F_GG = J_G.T @ J_G
        F_LL = J_L.T @ J_L
        F_GL = J_G.T @ J_L
        S_G  = F_GG - F_GL @ np.linalg.pinv(F_LL) @ F_GL.T
        eigvals = np.linalg.eigvalsh(S_G)
        self.assertTrue(np.all(eigvals >= -1e-10))


class TestSensitivityResultSchurAttachment(unittest.TestCase):
    """SensitivityResult.schur attribute."""

    def test_schur_none_by_default(self):
        J = np.eye(2)
        r = SensitivityResult(J, ["a", "b"], np.ones(2))
        self.assertIsNone(r.schur)

    def test_schur_stored_when_provided(self):
        J     = np.eye(3)
        schur = SchurResult(np.eye(1), ["global"], np.array([1.0]))
        r     = SensitivityResult(J, ["global", "l0", "l1"], np.ones(3), schur=schur)
        self.assertIs(r.schur, schur)

    def test_display_with_schur_runs(self):
        schur = SchurResult(np.eye(1), ["k"], np.array([0.5]))
        r = SensitivityResult(
            np.eye(2), ["k", "offset"], np.array([0.5, 1.0]), schur=schur
        )
        try:
            r.display()
        except Exception as e:
            self.fail(f"display() raised {e}")


class TestParameterEstimatorSchur(unittest.TestCase):
    """Integration: sensitivity() computes Schur for additive multi-exp model."""

    def setUp(self):
        self.est, self.true_x = _make_additive_multi_exp_estimator(n_exp=2)

    def test_sensitivity_returns_sensitivity_result(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertIsInstance(sens, SensitivityResult)

    def test_schur_is_not_none(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertIsNotNone(sens.schur)

    def test_schur_is_schur_result(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertIsInstance(sens.schur, SchurResult)

    def test_schur_global_param_count(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertEqual(len(sens.schur.param_names), 1)   # one global param

    def test_schur_fim_shape(self):
        sens = self.est.sensitivity(self.true_x)
        n_G  = len(self.est.global_parameters)
        self.assertEqual(sens.schur.fim.shape, (n_G, n_G))

    def test_schur_std_error_positive(self):
        sens = self.est.sensitivity(self.true_x)
        self.assertTrue(np.all(sens.schur.std_errors >= 0))

    def test_schur_condition_number_finite(self):
        """Additive model → S_G is well-conditioned."""
        sens = self.est.sensitivity(self.true_x)
        self.assertTrue(np.isfinite(sens.schur.condition_number))

    def test_schur_fim_leq_F_GG(self):
        """Max eigenvalue of S_G ≤ max eigenvalue of full F_GG block."""
        sens = self.est.sensitivity(self.true_x)
        n_G  = len(self.est.global_parameters)
        F_GG = sens.fim[:n_G, :n_G]
        ev_S   = np.linalg.eigvalsh(sens.schur.fim)
        ev_FGG = np.linalg.eigvalsh(F_GG)
        self.assertLessEqual(ev_S.max(), ev_FGG.max() + 1e-8)

    def test_schur_param_values_match_global_values(self):
        sens = self.est.sensitivity(self.true_x)
        n_G  = len(self.est.global_parameters)
        np.testing.assert_allclose(
            sens.schur.param_values,
            sens.param_values[:n_G],
            atol=1e-12,
        )

    def test_three_experiments_schur_present(self):
        est, true_x = _make_additive_multi_exp_estimator(n_exp=3)
        sens = est.sensitivity(true_x)
        self.assertIsNotNone(sens.schur)
        self.assertEqual(sens.schur.fim.shape, (1, 1))


class TestSingleExperimentNoLocals(unittest.TestCase):
    """Single-experiment estimator (no locals) → schur is None."""

    def test_schur_none_single_experiment(self):
        source = Source(func=lambda t: t)
        amp    = Amplifier(gain=1.0)
        scope  = Scope()
        sim = Simulation(
            [source, amp, scope],
            [Connection(source[0], amp[0]), Connection(amp[0], scope[0])],
            Solver=SSPRK22, dt=0.1, log=False,
        )
        t_meas = np.linspace(0.5, 3.0, 6)
        meas   = TimeSeriesData(time=t_meas, data=2.0 * t_meas)
        est = ParameterEstimator(simulator=sim)
        est.add_block_parameter(amp, "gain", value=1.0, bounds=(0.0, 10.0))
        est.add_timeseries(meas, signal=scope[0])
        sens = est.sensitivity(np.array([2.0]))
        self.assertIsNone(sens.schur)


class TestFitNestedNoGlobals(unittest.TestCase):
    """fit_nested() must raise ValueError when no global parameters are registered."""

    def test_raises_when_no_globals(self):
        source = Source(func=lambda t: t)
        amp    = Amplifier(gain=1.0)
        scope  = Scope()
        sim = Simulation(
            [source, amp, scope],
            [Connection(source[0], amp[0]), Connection(amp[0], scope[0])],
            Solver=SSPRK22, dt=0.1, log=False,
        )
        t_meas = np.linspace(0.5, 3.0, 6)
        meas   = TimeSeriesData(time=t_meas, data=2.0 * t_meas)
        est = ParameterEstimator(simulator=sim)
        est.add_local_block_parameter(
            0, "Amplifier", "gain", value=1.0, bounds=(0.0, 10.0)
        )
        est.add_timeseries(meas, signal=scope[0])

        with self.assertRaisesRegex(ValueError, "global"):
            est.fit_nested()


class TestSchurImport(unittest.TestCase):
    """SchurResult is importable from pathsim.opt."""

    def test_import(self):
        from pathsim.opt import SchurResult as SR
        self.assertIs(SR, SchurResult)


if __name__ == "__main__":
    unittest.main()
