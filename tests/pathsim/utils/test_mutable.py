########################################################################################
##
##                                  TESTS FOR
##                             'utils.mutable.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks._block import Block
from pathsim.blocks.lti import StateSpace
from pathsim.blocks.ctrl import PT1, PT2, LeadLag, PID, AntiWindupPID
from pathsim.blocks.lti import TransferFunctionNumDen, TransferFunctionZPG
from pathsim.blocks.filters import ButterworthLowpassFilter
from pathsim.blocks.sources import SinusoidalSource, ClockSource
from pathsim.blocks.delay import Delay
from pathsim.blocks.fir import FIR
from pathsim.blocks.samplehold import SampleHold

from pathsim.utils.mutable import mutable


# TESTS FOR DECORATOR ==================================================================

class TestMutableDecorator(unittest.TestCase):
    """Test the @mutable decorator mechanics."""

    def test_basic_construction(self):
        """Block construction should work as before."""
        pt1 = PT1(K=2.0, T=0.5)
        self.assertEqual(pt1.K, 2.0)
        self.assertEqual(pt1.T, 0.5)
        np.testing.assert_array_almost_equal(pt1.A, [[-2.0]])
        np.testing.assert_array_almost_equal(pt1.B, [[4.0]])

    def test_param_mutation_triggers_reinit(self):
        """Changing a mutable param should update derived state."""
        pt1 = PT1(K=1.0, T=1.0)
        np.testing.assert_array_almost_equal(pt1.A, [[-1.0]])
        np.testing.assert_array_almost_equal(pt1.B, [[1.0]])

        pt1.K = 5.0
        np.testing.assert_array_almost_equal(pt1.A, [[-1.0]])
        np.testing.assert_array_almost_equal(pt1.B, [[5.0]])

        pt1.T = 0.5
        np.testing.assert_array_almost_equal(pt1.A, [[-2.0]])
        np.testing.assert_array_almost_equal(pt1.B, [[10.0]])

    def test_batched_set(self):
        """set() should update multiple params with a single reinit."""
        pt1 = PT1(K=1.0, T=1.0)
        pt1.set(K=3.0, T=0.2)
        self.assertEqual(pt1.K, 3.0)
        self.assertEqual(pt1.T, 0.2)
        np.testing.assert_array_almost_equal(pt1.A, [[-5.0]])
        np.testing.assert_array_almost_equal(pt1.B, [[15.0]])

    def test_mutable_params_introspection(self):
        """_mutable_params should list all init params."""
        self.assertEqual(PT1._mutable_params, ("K", "T"))
        self.assertEqual(PT2._mutable_params, ("K", "T", "d"))
        self.assertEqual(PID._mutable_params, ("Kp", "Ki", "Kd", "f_max"))

    def test_mutable_params_inherited(self):
        """AntiWindupPID should accumulate parent and own params."""
        self.assertIn("Kp", AntiWindupPID._mutable_params)
        self.assertIn("Ks", AntiWindupPID._mutable_params)
        self.assertIn("limits", AntiWindupPID._mutable_params)
        # no duplicates
        self.assertEqual(
            len(AntiWindupPID._mutable_params),
            len(set(AntiWindupPID._mutable_params))
        )

    def test_no_reinit_during_construction(self):
        """Properties should not trigger reinit during __init__."""
        # If this doesn't hang or error, the init guard works
        pt1 = PT1(K=2.0, T=0.5)
        self.assertTrue(pt1._param_locked)


# TESTS FOR ENGINE PRESERVATION =========================================================

class TestEnginePreservation(unittest.TestCase):
    """Test that engine state is preserved across reinit."""

    def test_engine_preserved_same_dimension(self):
        """Engine should be preserved when state dimension doesn't change."""
        from pathsim.solvers.euler import EUF

        pt1 = PT1(K=1.0, T=1.0)
        pt1.set_solver(EUF, None)
        pt1.engine.state = np.array([42.0])

        # Mutate parameter
        pt1.K = 5.0

        # Engine should be preserved with same state
        self.assertIsNotNone(pt1.engine)
        np.testing.assert_array_equal(pt1.engine.state, [42.0])

    def test_engine_recreated_on_dimension_change(self):
        """Engine should be recreated when state dimension changes."""
        from pathsim.solvers.euler import EUF

        filt = ButterworthLowpassFilter(Fc=100, n=2)
        filt.set_solver(EUF, None)

        old_state_dim = len(filt.engine)
        self.assertEqual(old_state_dim, 2)

        # Change filter order -> dimension change
        filt.n = 4

        # Engine should exist but with new dimension
        self.assertIsNotNone(filt.engine)
        self.assertEqual(len(filt.engine), 4)


# TESTS FOR INHERITANCE =================================================================

class TestInheritance(unittest.TestCase):
    """Test that @mutable works with class hierarchies."""

    def test_antiwinduppid_construction(self):
        """AntiWindupPID should construct correctly with both decorators."""
        awpid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, Ks=10, limits=[-5, 5])
        self.assertEqual(awpid.Kp, 2)
        self.assertEqual(awpid.Ks, 10)

    def test_antiwinduppid_parent_param_mutation(self):
        """Mutating inherited param should reinit from most derived class."""
        awpid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, Ks=10, limits=[-5, 5])

        # Mutate inherited param
        awpid.Kp = 5.0

        # op_dyn should still be the antiwindup version (not plain PID)
        x = np.array([0.0, 0.0])
        u = np.array([1.0])
        result = awpid.op_dyn(x, u, 0)
        # For AntiWindupPID with these params, dx1 = f_max*(u-x1), dx2 = u - w
        self.assertEqual(len(result), 2)

    def test_antiwinduppid_own_param_mutation(self):
        """Mutating AntiWindupPID's own param should work."""
        awpid = AntiWindupPID(Kp=2, Ki=0.5, Kd=0.1, f_max=1e3, Ks=10, limits=[-5, 5])
        awpid.Ks = 20
        self.assertEqual(awpid.Ks, 20)


# TESTS FOR SPECIFIC BLOCKS =============================================================

class TestSpecificBlocks(unittest.TestCase):
    """Test @mutable on various block types."""

    def test_pt2(self):
        pt2 = PT2(K=1.0, T=1.0, d=0.5)
        A_before = pt2.A.copy()
        pt2.d = 0.7
        # A matrix should have changed
        self.assertFalse(np.allclose(pt2.A, A_before))

    def test_leadlag(self):
        ll = LeadLag(K=1.0, T1=0.5, T2=0.1)
        ll.K = 2.0
        self.assertEqual(ll.K, 2.0)
        # C and D should reflect new K
        np.testing.assert_array_almost_equal(ll.D, [[2.0 * 0.5 / 0.1]])

    def test_transfer_function_numden(self):
        tf = TransferFunctionNumDen(Num=[1], Den=[1, 1])
        np.testing.assert_array_almost_equal(tf.A, [[-1.0]])
        tf.Den = [1, 2]
        np.testing.assert_array_almost_equal(tf.A, [[-2.0]])

    def test_transfer_function_dimension_change(self):
        """Changing denominator order should change state dimension."""
        tf = TransferFunctionNumDen(Num=[1], Den=[1, 1])
        self.assertEqual(tf.A.shape, (1, 1))
        tf.Den = [1, 3, 2]  # second order
        self.assertEqual(tf.A.shape, (2, 2))

    def test_sinusoidal_source(self):
        s = SinusoidalSource(frequency=10, amplitude=2, phase=0.5)
        self.assertAlmostEqual(s._omega, 2*np.pi*10)
        s.frequency = 20
        self.assertAlmostEqual(s._omega, 2*np.pi*20)

    def test_delay(self):
        d = Delay(tau=0.01)
        self.assertEqual(d._buffer.delay, 0.01)
        d.tau = 0.05
        self.assertEqual(d._buffer.delay, 0.05)

    def test_clock_source(self):
        c = ClockSource(T=1.0, tau=0.0)
        self.assertEqual(c.events[0].t_period, 1.0)
        c.T = 2.0
        self.assertEqual(c.events[0].t_period, 2.0)

    def test_fir(self):
        f = FIR(coeffs=[0.5, 0.5], T=0.1)
        self.assertEqual(f.T, 0.1)
        f.T = 0.2
        self.assertEqual(f.T, 0.2)
        self.assertEqual(f.events[0].t_period, 0.2)

    def test_samplehold(self):
        sh = SampleHold(T=0.5, tau=0.0)
        sh.T = 1.0
        self.assertEqual(sh.T, 1.0)

    def test_butterworth_filter_mutation(self):
        filt = ButterworthLowpassFilter(Fc=100, n=2)
        A_before = filt.A.copy()
        filt.Fc = 200
        # Matrices should change
        self.assertFalse(np.allclose(filt.A, A_before))

    def test_butterworth_filter_order_change(self):
        filt = ButterworthLowpassFilter(Fc=100, n=2)
        self.assertEqual(filt.A.shape, (2, 2))
        filt.n = 4
        self.assertEqual(filt.A.shape, (4, 4))


# INTEGRATION TEST ======================================================================

class TestMutableInSimulation(unittest.TestCase):
    """Test parameter mutation in an actual simulation context."""

    def test_pt1_mutation_mid_simulation(self):
        """Mutating PT1 gain mid-simulation should affect output."""
        from pathsim import Simulation, Connection
        from pathsim.blocks.sources import Constant

        src = Constant(value=1.0)
        pt1 = PT1(K=1.0, T=0.1)

        sim = Simulation(
            blocks=[src, pt1],
            connections=[Connection(src, pt1)],
            dt=0.01
        )

        # Run for a bit
        sim.run(duration=1.0)
        output_before = pt1.outputs[0]

        # Change gain
        pt1.K = 5.0

        # Run more
        sim.run(duration=1.0)
        output_after = pt1.outputs[0]

        # With K=5 and enough settling time, output should approach 5.0
        self.assertGreater(output_after, output_before)


if __name__ == "__main__":
    unittest.main()
