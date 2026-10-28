########################################################################################
##
##                                     TESTS FOR
##                    sparse and matrix-free jacobians in the newton step
##                      ('optim/jacobian.py', 'optim/anderson.py')
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import scipy.sparse as sp

from pathsim.optim.anderson import NewtonAnderson, _factorize_sparse
from pathsim.optim.jacobian import NewtonOperator
from pathsim.optim.operator import Operator, DynamicOperator


# HELPERS ==============================================================================

def _linear_problem(n=6, seed=0):
    """Fixed point problem g(x) = J x + b with a small sparse jacobian J, where a
    newton step lands on the exact fixed point x* = (I - J)^-1 b."""
    rng = np.random.default_rng(seed)
    J = sp.random(n, n, density=0.3, random_state=seed, format="csr") * 0.4
    J = (J + sp.diags(-rng.uniform(0.5, 2.0, n))).tocsr()
    b = rng.normal(size=n)
    x_star = np.linalg.solve(np.eye(n) - J.toarray(), b)
    return J, b, x_star


# TESTS ================================================================================

class TestNewtonOperator(unittest.TestCase):

    def setUp(self):
        self.d = np.array([-1.0, -10.0, 3.0])
        self.op = NewtonOperator(
            newton_solve=lambda c, r: r / (c * self.d - 1.0),
            matvec=lambda v: self.d * v,
            shape=(3, 3)
            )

    def test_scaling_both_sides(self):
        self.assertEqual((0.5 * self.op).scale, 0.5)
        self.assertEqual((self.op * 0.5).scale, 0.5)
        self.assertEqual((2.0 * (0.5 * self.op)).scale, 1.0)
        self.assertEqual((-self.op).scale, -1.0)

    def test_numpy_scalar_scaling(self):
        #numpy scalars must defer to the reflected operator
        scaled = np.float64(0.1) * self.op
        self.assertIsInstance(scaled, NewtonOperator)
        self.assertAlmostEqual(scaled.scale, 0.1)

    def test_array_scaling_not_supported(self):
        with self.assertRaises(TypeError):
            self.op * np.ones(3)

    def test_newton_solve_uses_scale(self):
        r = np.array([1.0, 2.0, 3.0])
        c = 0.1
        y = (c * self.op).newton_solve(r)
        np.testing.assert_allclose((c * np.diag(self.d) - np.eye(3)) @ y, r)

    def test_matmul_and_toarray(self):
        v = np.array([1.0, -1.0, 2.0])
        np.testing.assert_allclose(self.op @ v, self.d * v)
        np.testing.assert_allclose((0.5 * self.op) @ v, 0.5 * self.d * v)
        np.testing.assert_allclose((0.5 * self.op).toarray(), 0.5 * np.diag(self.d))

    def test_matmul_without_matvec(self):
        op = NewtonOperator(newton_solve=lambda c, r: r)
        with self.assertRaises(NotImplementedError):
            op @ np.ones(2)

    def test_toarray_needs_shape(self):
        op = NewtonOperator(newton_solve=lambda c, r: r, matvec=lambda v: v)
        with self.assertRaises(ValueError):
            op.toarray()

    def test_ndim(self):
        self.assertEqual(self.op.ndim, 2)


class TestNewtonAnderson(unittest.TestCase):

    def test_sparse_matches_dense(self):
        J, b, x_star = _linear_problem()
        x = np.zeros(6)
        g = J @ x + b

        #the newton step of the optimizer lands on the exact fixed point of a linear problem
        xs, rs = NewtonAnderson()._newton(x, g, J)
        xd, rd = NewtonAnderson()._newton(x, g, J.toarray())
        np.testing.assert_allclose(xs, xd, atol=1e-12)
        np.testing.assert_allclose(xs, x_star, atol=1e-10)
        self.assertAlmostEqual(rs, rd)

        #and the full step (newton + anderson) agrees for sparse and dense
        ys, _ = NewtonAnderson().step(x, g, J)
        yd, _ = NewtonAnderson().step(x, g, J.toarray())
        np.testing.assert_allclose(ys, yd, atol=1e-12)

    def test_operator_matches_dense(self):
        J, b, x_star = _linear_problem()
        Jd = J.toarray()
        op = NewtonOperator(
            newton_solve=lambda c, r: np.linalg.solve(c * Jd - np.eye(6), r),
            matvec=lambda v: Jd @ v,
            shape=(6, 6)
            )
        x = np.zeros(6)
        g = J @ x + b

        xo, ro = NewtonAnderson()._newton(x, g, 1.0 * op)
        np.testing.assert_allclose(xo, x_star, atol=1e-10)

        yo, _ = NewtonAnderson().step(x, g, 1.0 * op)
        yd, _ = NewtonAnderson().step(x, g, Jd)
        np.testing.assert_allclose(yo, yd, atol=1e-10)

    def test_sparse_factorization_cached(self):
        J, b, _ = _linear_problem()
        opt = NewtonAnderson()
        x = np.zeros(6)

        opt.step(x, J @ x + b, J)
        lu = opt._lu
        self.assertIsNotNone(lu)

        #same jacobian -> same factorization
        opt.step(x + 0.1, J @ (x + 0.1) + b, J)
        self.assertIs(opt._lu, lu)

        #changed jacobian -> refactorized
        J2 = (J * 1.5).tocsr()
        opt.step(x, J2 @ x + b, J2)
        self.assertIsNot(opt._lu, lu)

    def test_switch_between_sparse_and_dense(self):
        J, b, x_star = _linear_problem()
        opt = NewtonAnderson()
        x = np.zeros(6)
        g = J @ x + b

        x1, _ = opt.step(x, g, J)
        opt.reset()
        x2, _ = opt.step(x, g, J.toarray())
        x3, _ = NewtonAnderson().step(x, g, J)
        np.testing.assert_allclose(x1, x2, atol=1e-12)
        np.testing.assert_allclose(x1, x3, atol=1e-12)

    def test_singular_sparse_newton_matrix(self):
        #J = I makes the newton matrix J - I exactly singular, no crash, finite result
        J = sp.identity(4, format="csr")
        x = np.ones(4)
        g = x + 0.5

        y, res = NewtonAnderson().step(x, g, J)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.isfinite(res))

    def test_solve_with_sparse_jacobian(self):
        J, b, x_star = _linear_problem()
        opt = NewtonAnderson()
        x, res, it = opt.solve(lambda x: J @ x + b - x, np.zeros(6), jac=lambda x: J, tolerance=1e-12)
        np.testing.assert_allclose(x, x_star, atol=1e-9)


class TestFactorizeSparse(unittest.TestCase):

    def test_regular_matrix(self):
        A = sp.csc_matrix(np.array([[2.0, 1.0], [0.0, 3.0]]))
        lu = _factorize_sparse(A)
        np.testing.assert_allclose(lu.solve(np.array([3.0, 3.0])), np.linalg.solve(A.toarray(), [3.0, 3.0]))

    def test_empty_and_zero_matrix(self):
        self.assertIsNone(_factorize_sparse(sp.csc_matrix((0, 0))))
        self.assertIsNone(_factorize_sparse(sp.csc_matrix((3, 3))))

    def test_exactly_singular_matrix(self):
        #structurally singular: a zero column, SuperLU raises
        A = sp.csc_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
        self.assertIsNone(_factorize_sparse(A))

    def test_numerically_singular_matrix(self):
        #rank deficient up to rounding: SuperLU does not raise, the pivot check rejects it
        A = sp.csc_matrix(np.array([[1.0, 1.0], [1.0, 1.0 + np.finfo(float).eps]]))
        self.assertIsNone(_factorize_sparse(A))

        #exactly rank deficient is rejected as well
        A = sp.csc_matrix(np.array([[1.0, 2.0], [2.0, 4.0]]))
        self.assertIsNone(_factorize_sparse(A))

    def test_nearly_singular_but_regular(self):
        A = sp.csc_matrix(np.diag([1.0, 1e-6]))
        self.assertIsNotNone(_factorize_sparse(A))


class TestOperatorSparse(unittest.TestCase):

    def test_sparse_linearization(self):
        A = sp.csr_matrix(np.array([[-1.0, 0.5], [0.0, -2.0]]))
        op = Operator(func=lambda x: A @ x, jac=lambda x: A)
        x0 = np.array([1.0, 2.0])
        op.linearize(x0)
        x1 = x0 + np.array([0.2, -0.1])
        np.testing.assert_allclose(op(x1), A @ x1, atol=1e-12)

    def test_operator_jacobian_linearization(self):
        Jd = np.array([[-1.0, 0.5], [0.0, -2.0]])
        nop = NewtonOperator(
            newton_solve=lambda c, r: np.linalg.solve(c * Jd - np.eye(2), r),
            matvec=lambda v: Jd @ v,
            shape=(2, 2)
            )
        op = Operator(func=lambda x: Jd @ x, jac=lambda x: nop)
        x0 = np.array([1.0, 2.0])
        op.linearize(x0)
        x1 = x0 + np.array([0.2, -0.1])
        np.testing.assert_allclose(op(x1), Jd @ x1, atol=1e-12)


class TestDynamicOperatorSparse(unittest.TestCase):

    def test_sparse_linearization(self):
        A = sp.csr_matrix(np.array([[-1.0, 0.5], [0.0, -2.0]]))
        Bm = sp.csr_matrix(np.array([[1.0], [0.0]]))
        op = DynamicOperator(
            func=lambda x, u, t: A @ x + Bm @ u,
            jac_x=lambda x, u, t: A,
            jac_u=lambda x, u, t: Bm
            )
        x0, u0 = np.array([1.0, 2.0]), np.array([0.5])
        op.linearize(x0, u0, 0.0)

        x1, u1 = x0 + np.array([0.1, -0.2]), u0 + 0.3
        expected = A @ x1 + Bm @ u1
        np.testing.assert_allclose(op(x1, u1, 0.0), np.asarray(expected).ravel(), atol=1e-12)

    def test_operator_jacobian_linearization(self):
        Jd = np.array([[-1.0, 0.5], [0.0, -2.0]])
        nop = NewtonOperator(
            newton_solve=lambda c, r: np.linalg.solve(c * Jd - np.eye(2), r),
            matvec=lambda v: Jd @ v,
            shape=(2, 2)
            )
        op = DynamicOperator(
            func=lambda x, u, t: Jd @ x,
            jac_x=lambda x, u, t: nop,
            jac_u=lambda x, u, t: np.zeros((2, 1))
            )
        x0 = np.array([1.0, 2.0])
        op.linearize(x0, np.zeros(1), 0.0)
        x1 = x0 + np.array([0.3, 0.1])
        np.testing.assert_allclose(op(x1, np.zeros(1), 0.0), Jd @ x1, atol=1e-12)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
