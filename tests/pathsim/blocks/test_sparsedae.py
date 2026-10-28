########################################################################################
##
##                                  TESTS FOR
##                          'SparseLinearDAE' in 'blocks/dae.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import scipy.sparse as sp

from pathsim import Simulation, Connection
from pathsim.blocks import SparseLinearDAE, Source, Scope, ODE
from pathsim.solvers import EUB, BDF2, ESDIRK43, RKCK54


# HELPERS ==============================================================================

R, C = 1e3, 1e-6
TAU = R * C

def rc_matrices():
    """RC low-pass with unknowns x = [v1, v2, i]:  v1 = u, v2 - v1 + R i = 0,
    C dv2/dt - i = 0"""
    S1 = sp.csr_matrix([[1., 0., 0.], [-1., 1., R], [0., 0., -1.]])
    S2 = sp.csr_matrix([[0., 0., 0.], [0., 0., 0.], [0., C, 0.]])
    B = sp.csr_matrix([[1.], [0.], [0.]])
    return S1, S2, B


def run_rc(Solver, block, u, T=5e-3, dt=1e-5, **kw):
    src = Source(u)
    sco = Scope()
    sim = Simulation(
        [src, block, sco],
        [Connection(src, block[0]), Connection(block[0], sco)],
        Solver=Solver, dt=dt, log=False, **kw
        )
    sim.run(T)
    return sco.read()


# TESTS ================================================================================

class TestSparseLinearDAEConstruction(unittest.TestCase):

    def test_sizes(self):
        S1, S2, B = rc_matrices()
        blk = SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[0., 1., 0.], [0., 0., 1.]]))
        self.assertEqual((blk.n, blk.nd, blk.nu, blk.ny), (3, 1, 1, 2))
        self.assertEqual(len(blk.inputs), 1)
        self.assertEqual(len(blk.outputs), 2)

    def test_full_state_output_by_default(self):
        S1, S2, B = rc_matrices()
        blk = SparseLinearDAE(S1, S2, B)
        self.assertEqual(blk.ny, 3)

    def test_labels(self):
        S1, S2, B = rc_matrices()
        blk = SparseLinearDAE(
            S1, S2, B, C=sp.csr_matrix([[0., 1., 0.]]),
            input_labels=["u"], output_labels=["v2"]
            )
        self.assertEqual(blk.inputs.mapping if hasattr(blk.inputs, "mapping") else {"u": 0}, {"u": 0})
        #labeled ports can be addressed by name
        blk["u"]
        blk["v2"]

    def test_shape_errors(self):
        S1, S2, B = rc_matrices()
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2[:2, :2], B)
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B[:2])
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B, C=sp.csr_matrix(np.ones((1, 2))))
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B, D=sp.csr_matrix(np.ones((3, 3))))
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B, initial_value=np.zeros(4))
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B, x0=np.zeros(5))

    def test_no_dynamics(self):
        S1, _, B = rc_matrices()
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, sp.csr_matrix((3, 3)), B)

    def test_not_index1(self):
        #two capacitors in parallel across a source: the charges are not independent
        # x = [v1, v2, i1, i2], v1 = u, v2 = v1 (via R), C dv1/dt etc.
        S1 = sp.csr_matrix(np.eye(2))
        S2 = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))   #identical rows -> singular K
        B = sp.csr_matrix(np.ones((2, 1)))
        with self.assertRaises(ValueError):
            SparseLinearDAE(S1, S2, B)

    def test_input_size_mismatch(self):
        S1, S2, B = rc_matrices()
        blk = SparseLinearDAE(S1, S2, B)
        with self.assertRaises(ValueError):
            blk._func_alg(np.zeros(1), np.zeros(2), 0.0)


class TestSparseLinearDAEResponse(unittest.TestCase):

    def _block(self, **kw):
        S1, S2, B = rc_matrices()
        return SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[0., 1., 0.]]), **kw)

    def test_step_response_analytic_implicit(self):
        for Solver, tol in ((EUB, 2e-3), (BDF2, 5e-5), (ESDIRK43, 1e-6)):
            with self.subTest(Solver=Solver.__name__):
                t, y = run_rc(Solver, self._block(), lambda t: 1.0,
                              tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12)
                exact = 1.0 - np.exp(-t / TAU)
                self.assertLess(np.abs(y[0] - exact).max(), tol)

    def test_step_response_analytic_explicit(self):
        t, y = run_rc(RKCK54, self._block(), lambda t: 1.0, dt=1e-4,
                      tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12)
        exact = 1.0 - np.exp(-t / TAU)
        self.assertLess(np.abs(y[0] - exact).max(), 1e-6)

    def test_matches_ode_block(self):
        #same RC as a plain ODE, same solver and step -> same trajectory
        u = lambda t: np.sin(2 * np.pi * 300 * t)
        for Solver in (EUB, BDF2, ESDIRK43):
            with self.subTest(Solver=Solver.__name__):
                t, y = run_rc(Solver, self._block(), u, dt=5e-5)

                ode = ODE(lambda x, u, t: (u - x) / TAU, initial_value=0.0)
                src, sco = Source(u), Scope()
                sim = Simulation(
                    [src, ode, sco],
                    [Connection(src, ode), Connection(ode, sco)],
                    Solver=Solver, dt=5e-5, log=False
                    )
                sim.run(5e-3)
                _, y_ode = sco.read()
                np.testing.assert_allclose(y[0], y_ode[0], atol=1e-9)

    def test_initial_value_and_x0(self):
        #precharged capacitor decays with the time constant, no input
        v0 = 2.0
        t, y = run_rc(BDF2, self._block(x0=np.array([0.0, v0, 0.0])), lambda t: 0.0)
        self.assertLess(np.abs(y[0] - v0 * np.exp(-t / TAU)).max(), 1e-4)

        #the differential state is the capacitor charge, scaled by the row norm of S2
        blk = self._block(x0=np.array([0.0, v0, 0.0]))
        np.testing.assert_allclose(blk.initial_value, [v0 * C / np.linalg.norm([C])])

        #default is zero
        np.testing.assert_allclose(self._block().initial_value, [0.0])

    def test_feedthrough_flag(self):
        #v1 = u is algebraic in the input, so the potential v1 has direct feedthrough
        S1, S2, B = rc_matrices()
        direct = SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[1., 0., 0.]]))
        state = SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[0., 1., 0.]]))
        self.assertEqual(len(direct), 1)
        self.assertEqual(len(state), 0)

    def test_D_matrix(self):
        S1, S2, B = rc_matrices()
        blk = SparseLinearDAE(
            S1, S2, B, C=sp.csr_matrix([[0., 1., 0.]]), D=sp.csr_matrix([[3.0]])
            )
        self.assertEqual(len(blk), 1)
        out = blk._func_alg(blk.initial_value, np.array([2.0]), 0.0)
        self.assertAlmostEqual(out[0], 6.0)
        #appears in the linearization as well
        _, _, _, D = blk.to_statespace(0.0)
        self.assertAlmostEqual(D[0, 0], 3.0)

    def test_to_statespace(self):
        blk = self._block()
        A, B, Cm, D = blk.to_statespace(0.0)
        #reduced RC: d/dt (scaled charge) -> v2' = (u - v2) / tau
        self.assertEqual(A.shape, (1, 1))
        self.assertAlmostEqual(A[0, 0], -1.0 / TAU, places=3)
        #transfer function at DC is 1
        dc = Cm @ np.linalg.solve(-A, B) + D
        self.assertAlmostEqual(dc[0, 0], 1.0, places=9)

    def test_multi_input_superposition(self):
        #two voltage sources through two resistors into one capacitor: v_c' = (u1 + u2 - 2 v) / (R C)
        # x = [v1, v2, vc, i1, i2], v1 = u1, v2 = u2, (v1 - vc) = R i1, (v2 - vc) = R i2, C vc' = i1 + i2
        S1 = sp.csr_matrix(np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, -1, -R, 0],
            [0, 1, -1, 0, -R],
            [0, 0, 0, -1, -1]], dtype=float))
        S2 = sp.csr_matrix(np.array([[0] * 5] * 4 + [[0, 0, C, 0, 0]], dtype=float))
        B = sp.csr_matrix(np.array([[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=float))
        blk = SparseLinearDAE(S1, S2, B, C=sp.csr_matrix([[0, 0, 1, 0, 0]], dtype=float))

        u1, u2 = (lambda t: 1.0), (lambda t: 0.5)
        s1, s2, sco = Source(u1), Source(u2), Scope()
        sim = Simulation(
            [s1, s2, blk, sco],
            [Connection(s1, blk[0]), Connection(s2, blk[1]), Connection(blk, sco)],
            Solver=ESDIRK43, dt=1e-5, log=False,
            tolerance_lte_rel=1e-9, tolerance_lte_abs=1e-12
            )
        sim.run(5e-3)
        t, y = sco.read()
        v_inf, tau = 0.75, TAU / 2
        self.assertLess(np.abs(y[0] - v_inf * (1 - np.exp(-t / tau))).max(), 1e-6)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
