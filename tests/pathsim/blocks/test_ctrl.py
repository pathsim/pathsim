########################################################################################
##
##                                  TESTS FOR
##                              'blocks.ctrl.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.ctrl import (
    PT1,
    PT2,
    LeadLag,
    PID,
    AntiWindupPID,
    RateLimiter
    )

#base solver for testing
from pathsim.solvers._solver import Solver

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestPT1(unittest.TestCase):
    """Test the implementation of the 'PT1' block class"""

    def test_init(self):

        pt1 = PT1(K=2.0, T=0.5)

        self.assertEqual(pt1.A.shape, (1, 1))
        self.assertEqual(pt1.B.shape, (1, 1))
        self.assertEqual(pt1.C.shape, (1, 1))
        self.assertEqual(pt1.D.shape, (1, 1))

        #check statespace matrices
        self.assertAlmostEqual(pt1.A[0, 0], -2.0)    # -1/T
        self.assertAlmostEqual(pt1.B[0, 0], 4.0)     # K/T
        self.assertAlmostEqual(pt1.C[0, 0], 1.0)
        self.assertAlmostEqual(pt1.D[0, 0], 0.0)


    def test_len(self):

        #PT1 has no direct passthrough (D=0)
        pt1 = PT1(K=2.0, T=0.5)
        self.assertEqual(len(pt1), 0)


    def test_shape(self):

        pt1 = PT1()
        self.assertEqual(pt1.shape, (1, 1))


    def test_set_solver(self):

        pt1 = PT1(K=1.0, T=1.0)
        pt1.set_solver(Solver, None)
        self.assertTrue(isinstance(pt1.engine, Solver))


    def test_embedding(self):

        #PT1 with D=0 -> output depends only on state, not input
        pt1 = PT1(K=1.0, T=1.0)
        pt1.set_solver(Solver, None)

        def src(t): return 1.0
        def ref(t): return 0.0  #initial state is zero, no passthrough

        E = Embedding(pt1, src, ref)
        self.assertEqual(*E.check_SISO(0))


class TestPT2(unittest.TestCase):
    """Test the implementation of the 'PT2' block class"""

    def test_init(self):

        pt2 = PT2(K=1.0, T=0.5, d=0.7)

        self.assertEqual(pt2.A.shape, (2, 2))
        self.assertEqual(pt2.B.shape, (2, 1))
        self.assertEqual(pt2.C.shape, (1, 2))
        self.assertEqual(pt2.D.shape, (1, 1))

        #check statespace matrices
        T, d, K = 0.5, 0.7, 1.0
        self.assertAlmostEqual(pt2.A[0, 0], 0.0)
        self.assertAlmostEqual(pt2.A[0, 1], 1.0)
        self.assertAlmostEqual(pt2.A[1, 0], -1.0 / T**2)
        self.assertAlmostEqual(pt2.A[1, 1], -2.0 * d / T)
        self.assertAlmostEqual(pt2.C[0, 0], K / T**2)


    def test_len(self):

        #PT2 has no direct passthrough (D=0)
        pt2 = PT2(K=1.0, T=1.0, d=0.5)
        self.assertEqual(len(pt2), 0)


    def test_shape(self):

        pt2 = PT2()
        self.assertEqual(pt2.shape, (1, 1))


    def test_damping_cases(self):

        #underdamped
        pt2 = PT2(K=1.0, T=1.0, d=0.3)
        self.assertEqual(pt2.A.shape, (2, 2))

        #critically damped
        pt2 = PT2(K=1.0, T=1.0, d=1.0)
        self.assertEqual(pt2.A.shape, (2, 2))

        #overdamped
        pt2 = PT2(K=1.0, T=1.0, d=2.0)
        self.assertEqual(pt2.A.shape, (2, 2))


class TestLeadLag(unittest.TestCase):
    """Test the implementation of the 'LeadLag' block class"""

    def test_init(self):

        ll = LeadLag(K=2.0, T1=0.5, T2=0.1)

        self.assertEqual(ll.A.shape, (1, 1))
        self.assertEqual(ll.B.shape, (1, 1))
        self.assertEqual(ll.C.shape, (1, 1))
        self.assertEqual(ll.D.shape, (1, 1))

        #check statespace matrices
        K, T1, T2 = 2.0, 0.5, 0.1
        self.assertAlmostEqual(ll.A[0, 0], -1.0 / T2)
        self.assertAlmostEqual(ll.B[0, 0], 1.0 / T2)
        self.assertAlmostEqual(ll.C[0, 0], K * (T2 - T1) / T2)
        self.assertAlmostEqual(ll.D[0, 0], K * T1 / T2)


    def test_len(self):

        #lead compensator: T1 > T2, has passthrough (D != 0)
        ll = LeadLag(K=1.0, T1=1.0, T2=0.5)
        self.assertEqual(len(ll), 1)

        #pure gain: T1 = T2, D = K
        ll = LeadLag(K=1.0, T1=1.0, T2=1.0)
        self.assertEqual(len(ll), 1)

        #T1 = 0: no passthrough (D = 0)
        ll = LeadLag(K=1.0, T1=0.0, T2=1.0)
        self.assertEqual(len(ll), 0)


    def test_shape(self):

        ll = LeadLag()
        self.assertEqual(ll.shape, (1, 1))


    def test_pure_gain(self):

        #when T1 = T2, should be pure gain K
        ll = LeadLag(K=3.0, T1=1.0, T2=1.0)
        self.assertAlmostEqual(ll.D[0, 0], 3.0)
        self.assertAlmostEqual(ll.C[0, 0], 0.0)


    def test_dc_gain(self):

        #DC gain should be K for any T1, T2
        #H(0) = K * (T1*0 + 1) / (T2*0 + 1) = K
        #In state space: -C*A^{-1}*B + D
        ll = LeadLag(K=2.5, T1=0.3, T2=0.8)
        dc_gain = -ll.C @ np.linalg.inv(ll.A) @ ll.B + ll.D
        self.assertAlmostEqual(dc_gain[0, 0], 2.5)


class TestPID(unittest.TestCase):
    """Test the implementation of the 'PID' block class"""

    def test_init(self):

        pid = PID(Kp=2, Ki=0.5, Kd=0.1, f_max=100)

        self.assertEqual(pid.A.shape, (2, 2))
        self.assertEqual(pid.B.shape, (2, 1))
        self.assertEqual(pid.C.shape, (1, 2))
        self.assertEqual(pid.D.shape, (1, 1))

        #check statespace matrices
        Kp, Ki, Kd, f_max = 2, 0.5, 0.1, 100
        self.assertAlmostEqual(pid.A[0, 0], -f_max)
        self.assertAlmostEqual(pid.A[0, 1], 0.0)
        self.assertAlmostEqual(pid.A[1, 0], 0.0)
        self.assertAlmostEqual(pid.A[1, 1], 0.0)
        self.assertAlmostEqual(pid.B[0, 0], f_max)
        self.assertAlmostEqual(pid.B[1, 0], 1.0)
        self.assertAlmostEqual(pid.C[0, 0], -Kd * f_max)
        self.assertAlmostEqual(pid.C[0, 1], Ki)
        self.assertAlmostEqual(pid.D[0, 0], Kd * f_max + Kp)


    def test_len(self):

        #has passthrough when Kp or Kd nonzero
        pid = PID(Kp=1, Ki=0, Kd=0)
        self.assertEqual(len(pid), 1)

        pid = PID(Kp=0, Ki=0, Kd=1)
        self.assertEqual(len(pid), 1)

        #pure integrator: no passthrough
        pid = PID(Kp=0, Ki=1, Kd=0)
        self.assertEqual(len(pid), 0)


    def test_shape(self):

        pid = PID()
        self.assertEqual(pid.shape, (1, 1))


    def test_set_solver(self):

        pid = PID(Kp=1, Ki=1, Kd=0.1)
        pid.set_solver(Solver, None)
        self.assertTrue(isinstance(pid.engine, Solver))


    def test_embedding(self):

        #PID with Kp=2, Ki=0, Kd=0 -> pure proportional, D=2
        pid = PID(Kp=2, Ki=0, Kd=0)
        pid.set_solver(Solver, None)

        def src(t): return 3.0
        def ref(t): return 6.0  #Kp * u = 2 * 3

        E = Embedding(pid, src, ref)
        self.assertAlmostEqual(*E.check_SISO(0), places=8)


class TestAntiWindupPID(unittest.TestCase):
    """Test the implementation of the 'AntiWindupPID' block class"""

    def test_init(self):

        pid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=100, Ks=10, limits=[-5, 5])

        self.assertEqual(pid.A.shape, (2, 2))
        self.assertEqual(pid.Ks, 10)
        self.assertEqual(pid.limits, [-5, 5])


    def test_len(self):

        pid = AntiWindupPID(Kp=1, Ki=0.5, Kd=0)
        self.assertEqual(len(pid), 1)


    def test_shape(self):

        pid = AntiWindupPID()
        self.assertEqual(pid.shape, (1, 1))


    def test_set_solver(self):

        pid = AntiWindupPID(Kp=1, Ki=1, Kd=0.1)
        pid.set_solver(Solver, None)
        self.assertTrue(isinstance(pid.engine, Solver))


class TestRateLimiter(unittest.TestCase):
    """Test the implementation of the 'RateLimiter' block class"""

    def test_init(self):

        rl = RateLimiter(rate=10.0, f_max=1e3)

        self.assertEqual(rl.rate, 10.0)
        self.assertEqual(rl.f_max, 1e3)
        self.assertEqual(rl.initial_value, 0.0)


    def test_len(self):

        #no direct passthrough
        rl = RateLimiter()
        self.assertEqual(len(rl), 0)


    def test_shape(self):

        rl = RateLimiter()
        self.assertEqual(rl.shape, (1, 1))


    def test_set_solver(self):

        rl = RateLimiter(rate=5.0)
        rl.set_solver(Solver, None)
        self.assertTrue(isinstance(rl.engine, Solver))


    def test_update(self):

        rl = RateLimiter(rate=1.0)
        rl.set_solver(Solver, None)

        #output should be engine state (initially 0)
        rl.update(0)
        self.assertAlmostEqual(rl.outputs[0], 0.0)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
