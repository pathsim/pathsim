########################################################################################
##
##                                     TESTS FOR
##                                 'optim/newton.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.newton import NewtonRaphson


# TESTS ================================================================================

class TestNewtonRaphson(unittest.TestCase):
    """
    Tests for the 'NewtonRaphson' root solver.
    """

    def test_init(self):
        nr = NewtonRaphson()
        self.assertTrue(nr.line_search)
        self.assertEqual(nr.iterations, 0)
        self.assertEqual(nr.residual, 0.0)
        self.assertTrue(bool(nr))

    def test_init_custom(self):
        nr = NewtonRaphson(tolerance=1e-6, iterations_max=50, beta=1e-3, line_search=False)
        self.assertEqual(nr.tolerance, 1e-6)
        self.assertEqual(nr.iterations_max, 50)
        self.assertEqual(nr.beta, 1e-3)
        self.assertFalse(nr.line_search)

    def test_scalar_root_with_jac(self):
        # x^2 - 2 = 0 -> sqrt(2)
        nr = NewtonRaphson(tolerance=1e-12)
        func = lambda x: x**2 - 2.0
        jac = lambda x: 2.0 * x
        x, res, it = nr.solve(func, np.array([1.0]), jac)
        self.assertAlmostEqual(x[0], np.sqrt(2.0), places=10)
        self.assertLess(res, 1e-12)

    def test_scalar_root_numerical_jac(self):
        # same problem but with finite-difference jacobian fallback
        nr = NewtonRaphson(tolerance=1e-10)
        func = lambda x: x**2 - 2.0
        x, res, it = nr.solve(func, np.array([1.0]))
        self.assertAlmostEqual(x[0], np.sqrt(2.0), places=8)

    def test_scalar_initial_value(self):
        # plain python float as initial value should also work
        nr = NewtonRaphson(tolerance=1e-12)
        func = lambda x: x**2 - 2.0
        jac = lambda x: 2.0 * x
        x, res, it = nr.solve(func, 1.0, jac)
        self.assertAlmostEqual(x[0], np.sqrt(2.0), places=10)

    def test_vector_linear_system(self):
        # A x = b -> residual A x - b, exact in one step
        A = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        func = lambda x: A @ x - b
        jac = lambda x: A
        nr = NewtonRaphson(tolerance=1e-12)
        x, res, it = nr.solve(func, np.zeros(2), jac)
        np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-10)
        self.assertLessEqual(it, 1)

    def test_vector_nonlinear_system(self):
        # intersection of circle and line:
        #   x0^2 + x1^2 = 1,  x1 = x0  ->  (1/sqrt2, 1/sqrt2)
        def func(x):
            return np.array([x[0]**2 + x[1]**2 - 1.0, x[1] - x[0]])
        nr = NewtonRaphson(tolerance=1e-12)
        x, res, it = nr.solve(func, np.array([0.5, 0.9]))
        np.testing.assert_allclose(x, [1/np.sqrt(2), 1/np.sqrt(2)], atol=1e-8)

    def test_warmstart_reduces_iterations(self):
        # solving from the solution should converge immediately
        nr = NewtonRaphson(tolerance=1e-12)
        func = lambda x: x**2 - 2.0
        jac = lambda x: 2.0 * x
        x, _, _ = nr.solve(func, np.array([1.0]), jac)
        _, _, it_warm = nr.solve(func, x, jac)
        self.assertEqual(it_warm, 0)

    def test_fd_step_near_zero_start(self):
        # finite-difference jacobian must stay meaningful when the start is
        # exactly zero: F(x) = x - 3, x0 = 0 (relative step alone would vanish)
        nr = NewtonRaphson(tolerance=1e-12)
        func = lambda x: x - 3.0
        x, res, it = nr.solve(func, np.array([0.0]))
        self.assertAlmostEqual(x[0], 3.0, places=10)
        self.assertLess(res, 1e-12)

    def test_line_search_global_convergence(self):
        # stiff residual where an undamped full step overshoots badly
        def func(x):
            return np.array([np.exp(x[0]) - 2.0])
        nr = NewtonRaphson(tolerance=1e-10, line_search=True)
        x, res, it = nr.solve(func, np.array([8.0]))
        self.assertAlmostEqual(x[0], np.log(2.0), places=7)

    def test_diagnostics_updated(self):
        nr = NewtonRaphson(tolerance=1e-12)
        func = lambda x: x**2 - 2.0
        jac = lambda x: 2.0 * x
        x, res, it = nr.solve(func, np.array([1.0]), jac)
        self.assertEqual(nr.iterations, it)
        self.assertEqual(nr.residual, res)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
