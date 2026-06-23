########################################################################################
##
##                                     TESTS FOR 
##                                'optim/anderson.py'
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.optim.anderson import (
    Anderson,
    NewtonAnderson,
    solve_root
    )


class TestAnderson(unittest.TestCase):
    """
    Extended tests for the 'Anderson' class.
    """

    def test_init(self):
        m = 5
        aa = Anderson(m)
        self.assertEqual(aa.m, m)
        self.assertFalse(aa.restart)
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)
        self.assertIsNone(aa.x_prev)
        self.assertIsNone(aa.r_prev)

    def test_reset(self):
        aa = Anderson(5)
        # artificially add some entries
        aa.dx_buffer.append(np.array([1.0]))
        aa.dr_buffer.append(np.array([2.0]))
        aa.x_prev = np.array([1.0])
        aa.r_prev = np.array([2.0])
        aa.reset()
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)
        self.assertIsNone(aa.x_prev)
        self.assertIsNone(aa.r_prev)

    def test_step_scalar(self):
        aa = Anderson(2)
        x, g = 1.0, 2.0
        result, residual = aa.step(x, g)
        self.assertEqual(result, g)
        self.assertEqual(residual, abs(g - x))

    def test_step_vector(self):
        aa = Anderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = aa.step(x, g)
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_solve_converge_scalar(self):
        # Solve x = cos(x) using solve method
        def func_scalar(x):
            return np.cos(x) - x  # f(x)=0 => x=cos(x)
        aa = Anderson(m=5)
        x0 = np.array([0.0])
        x_sol, res, iters = aa.solve(func_scalar, x0, iterations_max=200, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 0.7390851332151607, places=7)

    def test_solve_converge_vector(self):
        # Solve system: x = x / ||x||, so solution is any unit vector. Start from random point.
        def func_vec(x):
            norm = np.linalg.norm(x)
            return x / norm - x
        aa = Anderson(m=5)
        x0 = np.array([1.0, 1.0])
        x_sol, res, iters = aa.solve(func_vec, x0, iterations_max=200, tolerance=1e-10)
        # Check unit circle solution
        self.assertAlmostEqual(np.linalg.norm(x_sol), 1.0, places=7)

    def test_restart_behavior(self):
        # Check if restart clears the buffers after they are full
        aa = Anderson(m=2, restart=True)
        x = np.array([1.0, 2.0])
        g = np.array([1.5, 2.5])
        # step 1
        aa.step(x, g)
        # step 2 (fills buffer)
        x, res = aa.step(g, g+0.1)
        # step 3 (trigger restart)
        x, res = aa.step(x, x+0.2)
        self.assertEqual(len(aa.dx_buffer), 0)
        self.assertEqual(len(aa.dr_buffer), 0)


class TestNewtonAnderson(unittest.TestCase):
    """
    Extended tests for the 'NewtonAnderson' class.
    """

    def test_init(self):
        m = 5
        naa = NewtonAnderson(m)
        self.assertEqual(naa.m, m)
        self.assertFalse(naa.restart)
        self.assertEqual(len(naa.dx_buffer), 0)
        self.assertEqual(len(naa.dr_buffer), 0)

    def test_step_no_jacobian(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 3.0])
        result, residual = naa.step(x, g)
        # same behavior as Anderson if no jac
        np.testing.assert_array_equal(result, g)
        self.assertAlmostEqual(residual, np.linalg.norm(g - x))

    def test_step_with_jacobian_scalar(self):
        # Solve a scalar equation quickly:
        # Suppose g(x)=cos(x), jac= -sin(x)
        naa = NewtonAnderson(2)
        x = 0.0
        g = np.cos(x)
        j = -np.sin(x)
        result, residual = naa.step(x, g, j)
        # Just ensure it runs without error and returns valid result
        # Result is now a 1D array of length 1 due to flatten()
        self.assertTrue(np.isscalar(result) or (isinstance(result, np.ndarray) and result.size == 1))
        self.assertTrue(np.isscalar(residual))

    def test_step_with_jacobian_vector(self):
        naa = NewtonAnderson(2)
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 4.0])
        # jac of g(x)= [2,0;0,2] for a trivial linear system
        jac = np.array([[2.0, 0.0],[0.0, 2.0]])
        result, residual = naa.step(x, g, jac)
        self.assertEqual(result.shape, (2,))
        self.assertTrue(residual >= 0)

    def test_solve_scalar_equation(self):
        # Solve x = cos(x)
        naa = NewtonAnderson(m=5)
        def func_scalar(x):
            return np.cos(x) - x
        def jac_scalar(x):
            return -np.sin(x) - 1.0  # derivative of (cos(x)-x)
        x0 = np.array([0.0])
        x_sol, res, iters = naa.solve(func_scalar, x0, jac=jac_scalar, iterations_max=200, tolerance=1e-10)
        self.assertAlmostEqual(x_sol[0], 0.7390851332151607, places=7)

    def test_newton_lu_matches_direct_solve(self):
        # the cached LU path must match a direct solve of the Newton matrix (jac - I)
        naa = NewtonAnderson()
        x = np.array([1.0, -2.0, 0.5])
        g = np.array([0.3, 0.1, -0.4])
        jac = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 2.0]])
        y, _ = naa._newton(x, g, jac)
        expected = x - np.linalg.solve(jac - np.eye(3), g - x)
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_newton_lu_cache_reuse(self):
        # unchanged Newton matrix -> factorization is reused, changed -> refactored
        naa = NewtonAnderson()
        x = np.array([1.0, 2.0])
        g = np.array([2.0, 4.0])
        jac = np.array([[2.0, 0.0], [0.0, 3.0]])

        naa._newton(x, g, jac)
        cached = naa._lu
        self.assertIsNotNone(cached)

        #same matrix -> same factorization object
        naa._newton(x, g, jac)
        self.assertIs(naa._lu, cached)

        #different matrix -> new factorization
        naa._newton(x, g, jac + np.eye(2))
        self.assertIsNot(naa._lu, cached)

    def test_newton_cache_cleared_on_reset(self):
        naa = NewtonAnderson()
        jac = np.array([[2.0, 0.0], [0.0, 3.0]])
        naa._newton(np.array([1.0, 2.0]), np.array([2.0, 4.0]), jac)
        self.assertIsNotNone(naa._A)
        naa.reset()
        self.assertIsNone(naa._A)
        self.assertIsNone(naa._lu)


class TestSolveRoot(unittest.TestCase):
    """
    Tests for the 'solve_root' Anderson-accelerated damped Newton root solver.
    """

    def test_scalar_with_jac(self):
        # x^2 - 2 = 0 -> sqrt(2)
        x, res, it = solve_root(NewtonAnderson(), lambda x: x**2 - 2.0,
                                np.array([1.0]), jac=lambda x: 2.0*x, tolerance=1e-12)
        self.assertAlmostEqual(x[0], np.sqrt(2.0), places=10)
        self.assertLess(res, 1e-12)

    def test_scalar_numerical_jac(self):
        x, res, it = solve_root(NewtonAnderson(), lambda x: x**2 - 2.0,
                                np.array([1.0]), tolerance=1e-10)
        self.assertAlmostEqual(x[0], np.sqrt(2.0), places=8)

    def test_basin_preservation_cold_start(self):
        # x^2 - 4 = 0 has roots +-2; the damped Newton must stay in the basin
        # of the start, not jump across the root (the bug a raw fixed-point map
        # x + F would cause)
        func = lambda x: x**2 - 4.0
        x_pos, *_ = solve_root(NewtonAnderson(), func, np.array([1.0]), tolerance=1e-12)
        x_neg, *_ = solve_root(NewtonAnderson(), func, np.array([-1.0]), tolerance=1e-12)
        self.assertAlmostEqual(x_pos[0], 2.0, places=8)
        self.assertAlmostEqual(x_neg[0], -2.0, places=8)

    def test_near_zero_start(self):
        # F(x) = x - 3, x0 = 0 -> 3 (relies on the num_jac absolute step floor)
        x, res, it = solve_root(NewtonAnderson(), lambda x: x - 3.0,
                                np.array([0.0]), tolerance=1e-12)
        self.assertAlmostEqual(x[0], 3.0, places=10)

    def test_vector_nonlinear(self):
        # circle and line: x0^2 + x1^2 = 1, x1 = x0 -> (1/sqrt2, 1/sqrt2)
        def func(x):
            return np.array([x[0]**2 + x[1]**2 - 1.0, x[1] - x[0]])
        x, res, it = solve_root(NewtonAnderson(), func, np.array([0.5, 0.9]), tolerance=1e-12)
        np.testing.assert_allclose(x, [1/np.sqrt(2), 1/np.sqrt(2)], atol=1e-8)

    def test_warmstart_converges_immediately(self):
        func = lambda x: x**2 - 2.0
        x, *_ = solve_root(NewtonAnderson(), func, np.array([1.0]), tolerance=1e-12)
        _, _, it = solve_root(NewtonAnderson(), func, x, tolerance=1e-12)
        self.assertEqual(it, 0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)