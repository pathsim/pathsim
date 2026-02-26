########################################################################################
##
##                                  TESTS FOR
##                              'blocks.logic.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.logic import (
    GreaterThan,
    LessThan,
    Equal,
    LogicAnd,
    LogicOr,
    LogicNot,
)

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestGreaterThan(unittest.TestCase):
    """
    Test the implementation of the 'GreaterThan' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = GreaterThan()

        #test a > b
        def src(t): return t, 5.0
        def ref(t): return float(t > 5.0)
        E = Embedding(B, src, ref)

        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))

    def test_equal_values(self):
        """test that equal values return 0"""

        B = GreaterThan()
        B.inputs[0] = 5.0
        B.inputs[1] = 5.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)

    def test_less_than(self):
        """test that a < b returns 0"""

        B = GreaterThan()
        B.inputs[0] = 3.0
        B.inputs[1] = 5.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)


class TestLessThan(unittest.TestCase):
    """
    Test the implementation of the 'LessThan' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = LessThan()

        #test a < b
        def src(t): return t, 5.0
        def ref(t): return float(t < 5.0)
        E = Embedding(B, src, ref)

        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))

    def test_equal_values(self):
        """test that equal values return 0"""

        B = LessThan()
        B.inputs[0] = 5.0
        B.inputs[1] = 5.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)


class TestEqual(unittest.TestCase):
    """
    Test the implementation of the 'Equal' block class
    """

    def test_equal(self):
        """test that equal values return 1"""

        B = Equal()
        B.inputs[0] = 5.0
        B.inputs[1] = 5.0
        B.update(0)
        self.assertEqual(B.outputs[0], 1.0)

    def test_not_equal(self):
        """test that different values return 0"""

        B = Equal()
        B.inputs[0] = 5.0
        B.inputs[1] = 6.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)

    def test_tolerance(self):
        """test tolerance parameter"""

        B = Equal(tolerance=0.1)
        B.inputs[0] = 5.0
        B.inputs[1] = 5.05
        B.update(0)
        self.assertEqual(B.outputs[0], 1.0)

        B.inputs[1] = 5.2
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)


class TestLogicAnd(unittest.TestCase):
    """
    Test the implementation of the 'LogicAnd' block class
    """

    def test_truth_table(self):
        """test all combinations of boolean inputs"""

        B = LogicAnd()

        cases = [
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ]

        for a, b, expected in cases:
            B.inputs[0] = a
            B.inputs[1] = b
            B.update(0)
            self.assertEqual(B.outputs[0], expected, f"AND({a}, {b}) should be {expected}")

    def test_nonzero_is_true(self):
        """test that nonzero values are treated as true"""

        B = LogicAnd()
        B.inputs[0] = 5.0
        B.inputs[1] = -3.0
        B.update(0)
        self.assertEqual(B.outputs[0], 1.0)


class TestLogicOr(unittest.TestCase):
    """
    Test the implementation of the 'LogicOr' block class
    """

    def test_truth_table(self):
        """test all combinations of boolean inputs"""

        B = LogicOr()

        cases = [
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ]

        for a, b, expected in cases:
            B.inputs[0] = a
            B.inputs[1] = b
            B.update(0)
            self.assertEqual(B.outputs[0], expected, f"OR({a}, {b}) should be {expected}")


class TestLogicNot(unittest.TestCase):
    """
    Test the implementation of the 'LogicNot' block class
    """

    def test_true_to_false(self):
        """test that nonzero input gives 0"""

        B = LogicNot()
        B.inputs[0] = 1.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)

    def test_false_to_true(self):
        """test that zero input gives 1"""

        B = LogicNot()
        B.inputs[0] = 0.0
        B.update(0)
        self.assertEqual(B.outputs[0], 1.0)

    def test_nonzero_is_true(self):
        """test that arbitrary nonzero values are treated as true"""

        B = LogicNot()
        B.inputs[0] = 42.0
        B.update(0)
        self.assertEqual(B.outputs[0], 0.0)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
