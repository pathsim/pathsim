########################################################################################
##
##                     Testing logic and comparison block systems
##
##   Verifies comparison (GreaterThan, LessThan, Equal) and boolean logic
##   (LogicAnd, LogicOr, LogicNot) blocks in full simulation context.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    Constant,
    Scope,
    )

from pathsim.blocks.logic import (
    GreaterThan,
    LessThan,
    Equal,
    LogicAnd,
    LogicOr,
    LogicNot,
    )


# TESTCASE =============================================================================

class TestComparisonSystem(unittest.TestCase):
    """
    Test comparison blocks in a simulation that compares a sine wave
    against a constant threshold.

    System: Source(sin(t)) → GT/LT/EQ → Scope
            Constant(0)   ↗

    Verify: GT outputs 1 when sin(t) > 0, LT outputs 1 when sin(t) < 0
    """

    def setUp(self):

        Src = Source(lambda t: np.sin(2 * np.pi * t))
        Thr = Constant(0.0)

        self.GT = GreaterThan()
        self.LT = LessThan()

        self.Sco = Scope(labels=["signal", "gt_zero", "lt_zero"])

        blocks = [Src, Thr, self.GT, self.LT, self.Sco]

        connections = [
            Connection(Src, self.GT["a"], self.LT["a"], self.Sco[0]),
            Connection(Thr, self.GT["b"], self.LT["b"]),
            Connection(self.GT, self.Sco[1]),
            Connection(self.LT, self.Sco[2]),
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_gt_lt_complementary(self):
        """GT and LT should be complementary (sum to 1) away from zero crossings"""

        self.Sim.run(duration=3.0, reset=True)

        time, [sig, gt, lt] = self.Sco.read()

        #away from zero crossings, GT + LT should be 1 (exactly one is true)
        mask = np.abs(sig) > 0.1
        result = gt[mask] + lt[mask]

        self.assertTrue(np.allclose(result, 1.0),
            "GT and LT should be complementary away from zero crossings")


    def test_gt_matches_positive(self):
        """GT output should be 1 when signal is clearly positive"""

        self.Sim.run(duration=3.0, reset=True)

        time, [sig, gt, lt] = self.Sco.read()

        mask_pos = sig > 0.2
        self.assertTrue(np.all(gt[mask_pos] == 1.0),
            "GT should be 1 when signal is positive")

        mask_neg = sig < -0.2
        self.assertTrue(np.all(gt[mask_neg] == 0.0),
            "GT should be 0 when signal is negative")


class TestLogicGateSystem(unittest.TestCase):
    """
    Test logic gates combining two comparison outputs.

    System: Two sine waves at different frequencies compared against 0,
    then combined with AND/OR/NOT.

    Verify: Logic truth tables hold across the simulation.
    """

    def setUp(self):

        #two signals with different frequencies so they go in and out of phase
        Src1 = Source(lambda t: np.sin(2 * np.pi * 1.0 * t))
        Src2 = Source(lambda t: np.sin(2 * np.pi * 1.5 * t))
        Zero = Constant(0.0)

        GT1 = GreaterThan()
        GT2 = GreaterThan()

        self.AND = LogicAnd()
        self.OR = LogicOr()
        self.NOT = LogicNot()

        self.Sco = Scope(labels=["gt1", "gt2", "and", "or", "not1"])

        blocks = [Src1, Src2, Zero, GT1, GT2,
                  self.AND, self.OR, self.NOT, self.Sco]

        connections = [
            Connection(Src1, GT1["a"]),
            Connection(Src2, GT2["a"]),
            Connection(Zero, GT1["b"], GT2["b"]),
            Connection(GT1, self.AND["a"], self.OR["a"], self.NOT, self.Sco[0]),
            Connection(GT2, self.AND["b"], self.OR["b"], self.Sco[1]),
            Connection(self.AND, self.Sco[2]),
            Connection(self.OR, self.Sco[3]),
            Connection(self.NOT, self.Sco[4]),
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_and_gate(self):
        """AND should only be 1 when both inputs are 1"""

        self.Sim.run(duration=5.0, reset=True)

        time, [gt1, gt2, and_out, or_out, not_out] = self.Sco.read()

        #where both are 1, AND should be 1
        both_true = (gt1 == 1.0) & (gt2 == 1.0)
        if np.any(both_true):
            self.assertTrue(np.all(and_out[both_true] == 1.0))

        #where either is 0, AND should be 0
        either_false = (gt1 == 0.0) | (gt2 == 0.0)
        if np.any(either_false):
            self.assertTrue(np.all(and_out[either_false] == 0.0))


    def test_or_gate(self):
        """OR should be 1 when either input is 1"""

        self.Sim.run(duration=5.0, reset=True)

        time, [gt1, gt2, and_out, or_out, not_out] = self.Sco.read()

        #where both are 0, OR should be 0
        both_false = (gt1 == 0.0) & (gt2 == 0.0)
        if np.any(both_false):
            self.assertTrue(np.all(or_out[both_false] == 0.0))

        #where either is 1, OR should be 1
        either_true = (gt1 == 1.0) | (gt2 == 1.0)
        if np.any(either_true):
            self.assertTrue(np.all(or_out[either_true] == 1.0))


    def test_not_gate(self):
        """NOT should invert its input"""

        self.Sim.run(duration=5.0, reset=True)

        time, [gt1, gt2, and_out, or_out, not_out] = self.Sco.read()

        #NOT should be inverse of GT1
        self.assertTrue(np.allclose(not_out + gt1, 1.0),
            "NOT should invert its input")


class TestEqualSystem(unittest.TestCase):
    """
    Test Equal block detecting when two signals are close.

    System: Source(sin(t)) → Equal ← Source(sin(t + small_offset))
    """

    def test_equal_detects_match(self):
        """Equal should output 1 when signals match within tolerance"""

        Src1 = Constant(3.14)
        Src2 = Constant(3.14)

        Eq = Equal(tolerance=0.01)
        Sco = Scope()

        Sim = Simulation(
            blocks=[Src1, Src2, Eq, Sco],
            connections=[
                Connection(Src1, Eq["a"]),
                Connection(Src2, Eq["b"]),
                Connection(Eq, Sco),
                ],
            dt=0.1,
            log=False
            )

        Sim.run(duration=1.0, reset=True)

        time, [eq_out] = Sco.read()

        self.assertTrue(np.all(eq_out == 1.0),
            "Equal should output 1 for identical signals")


    def test_equal_detects_mismatch(self):
        """Equal should output 0 when signals differ"""

        Src1 = Constant(1.0)
        Src2 = Constant(2.0)

        Eq = Equal(tolerance=0.01)
        Sco = Scope()

        Sim = Simulation(
            blocks=[Src1, Src2, Eq, Sco],
            connections=[
                Connection(Src1, Eq["a"]),
                Connection(Src2, Eq["b"]),
                Connection(Eq, Sco),
                ],
            dt=0.1,
            log=False
            )

        Sim.run(duration=1.0, reset=True)

        time, [eq_out] = Sco.read()

        self.assertTrue(np.all(eq_out == 0.0),
            "Equal should output 0 for different signals")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
