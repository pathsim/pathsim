########################################################################################
##
##                                  TESTS FOR 
##                               'blocks.math.py'
##
########################################################################################

# IMPORTS ==============================================================================
import unittest
import numpy as np
from pathsim.blocks.math import *
from tests.pathsim.blocks._embedding import Embedding

# TESTS ================================================================================

class TestSin(unittest.TestCase):
    """
    Test the implementation of the 'Sin' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Sin()

        #test embedding of block with SISO
        def src(t): return t
        def ref(t): return np.sin(t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t, 5*t
        def ref(t): return np.sin(t), np.sin(5*t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestCos(unittest.TestCase):
    """
    Test the implementation of the 'Cos' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Cos()

        #test embedding of block with SISO
        def src(t): return t
        def ref(t): return np.cos(t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t, 3*t
        def ref(t): return np.cos(t), np.cos(3*t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestSqrt(unittest.TestCase):
    """
    Test the implementation of the 'Sqrt' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Sqrt()

        #test embedding of block with SISO
        def src(t): return abs(t) + 1  # Ensure positive input
        def ref(t): return np.sqrt(abs(abs(t) + 1)) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return abs(t) + 1, abs(2*t) + 1
        def ref(t): return np.sqrt(abs(abs(t) + 1)), np.sqrt(abs(abs(2*t) + 1))
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestAbs(unittest.TestCase):
    """
    Test the implementation of the 'Abs' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Abs()

        #test embedding of block with SISO
        def src(t): return t - 5  # Test with negative values
        def ref(t): return abs(t - 5) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t - 5, -2*t + 3
        def ref(t): return abs(t - 5), abs(-2*t + 3)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestPow(unittest.TestCase):
    """
    Test the implementation of the 'Pow' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        # Test with default exponent (2)
        B = Pow()
        def src(t): return t + 1
        def ref(t): return np.power(t + 1, 2) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))
        
        # Test with custom exponent (3)
        B = Pow(exponent=3)
        def src(t): return t + 1, 2*t + 1
        def ref(t): return np.power(t + 1, 3), np.power(2*t + 1, 3)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestPowProd(unittest.TestCase):
    """
    Test the implementation of the 'PowProd' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""
        
        # Test with default exponent (2) - computes u1^2 * u2^2 * ...
        B = PowProd()
        def src(t): return t + 1, t + 2  # Ensure positive inputs
        def ref(t): 
            u1, u2 = t + 1, t + 2
            return np.prod([u1**2, u2**2])
        E = Embedding(B, src, ref)
        
        for t in range(1, 10): self.assertTrue(np.allclose(*E.check_MIMO(t)))
        
        # Test with custom exponents - different power for each input
        B = PowProd(exponents=[2, 3, 1])
        def src(t): return t + 1, t + 2, t + 3
        def ref(t): 
            u1, u2, u3 = t + 1, t + 2, t + 3
            return np.prod([u1**2, u2**3, u3**1])
        E = Embedding(B, src, ref)
        
        for t in range(1, 10): self.assertTrue(np.allclose(*E.check_MIMO(t)))
        
        # Test with scalar exponent applied to multiple inputs
        B = PowProd(exponents=3)
        def src(t): return t + 1, t + 2
        def ref(t): 
            u1, u2 = t + 1, t + 2
            return np.prod([u1**3, u2**3])
        E = Embedding(B, src, ref)
        
        for t in range(1, 10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestExp(unittest.TestCase):
    """
    Test the implementation of the 'Exp' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Exp()

        #test embedding of block with SISO
        def src(t): return t * 0.1  # Small values to avoid overflow
        def ref(t): return np.exp(t * 0.1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t * 0.1, t * 0.05
        def ref(t): return np.exp(t * 0.1), np.exp(t * 0.05)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestLog(unittest.TestCase):
    """
    Test the implementation of the 'Log' block class
    """
    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Log()

        #test embedding of block with SISO
        def src(t): return t + 1  # Ensure positive input
        def ref(t): return np.log(t + 1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t + 1, 2*t + 1
        def ref(t): return np.log(t + 1), np.log(2*t + 1)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestLog10(unittest.TestCase):
    """
    Test the implementation of the 'Log10' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Log10()

        #test embedding of block with SISO
        def src(t): return t + 1  # Ensure positive input
        def ref(t): return np.log10(t + 1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t + 1, 3*t + 1
        def ref(t): return np.log10(t + 1), np.log10(3*t + 1)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestTan(unittest.TestCase):
    """
    Test the implementation of the 'Tan' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Tan()

        #test embedding of block with SISO
        def src(t): return t * 0.1  # Small values to avoid singularities
        def ref(t): return np.tan(t * 0.1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t * 0.1, t * 0.05
        def ref(t): return np.tan(t * 0.1), np.tan(t * 0.05)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestSinh(unittest.TestCase):
    """
    Test the implementation of the 'Sinh' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Sinh()

        #test embedding of block with SISO
        def src(t): return t * 0.1
        def ref(t): return np.sinh(t * 0.1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t * 0.1, t * 0.2
        def ref(t): return np.sinh(t * 0.1), np.sinh(t * 0.2)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestCosh(unittest.TestCase):
    """
    Test the implementation of the 'Cosh' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Cosh()

        #test embedding of block with SISO
        def src(t): return t * 0.1
        def ref(t): return np.cosh(t * 0.1) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t * 0.1, t * 0.15
        def ref(t): return np.cosh(t * 0.1), np.cosh(t * 0.15)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestTanh(unittest.TestCase):
    """
    Test the implementation of the 'Tanh' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Tanh()

        #test embedding of block with SISO
        def src(t): return t
        def ref(t): return np.tanh(t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t, 2*t
        def ref(t): return np.tanh(t), np.tanh(2*t)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestAtan(unittest.TestCase):
    """
    Test the implementation of the 'Atan' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Atan()

        #test embedding of block with SISO
        def src(t): return t
        def ref(t): return np.arctan(t) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        #test embedding of block with MIMO
        def src(t): return t, 3*t
        def ref(t): return np.arctan(t), np.arctan(3*t)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestNorm(unittest.TestCase):
    """
    Test the implementation of the 'Norm' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        B = Norm()

        #test embedding of block with SISO
        def src(t): return t + 1
        def ref(t): return np.linalg.norm(t + 1) 
        E = Embedding(B, src, ref)
        
        for t in range(1, 10): self.assertEqual(*E.check_SISO(t))  # Start from 1 to avoid zero

        #test embedding of block with MIMO (vector norm)
        def src(t): return t + 1, 2*t + 1
        def ref(t): return np.linalg.norm([t + 1, 2*t + 1])
        E = Embedding(B, src, ref)
        
        for t in range(1, 10): self.assertTrue(np.allclose(*E.check_MIMO(t))) 


class TestMod(unittest.TestCase):
    """
    Test the implementation of the 'Mod' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""

        # Test with default modulus (1.0)
        B = Mod()
        def src(t): return t * 0.7  # Values that will wrap around
        def ref(t): return np.mod(t * 0.7, 1.0) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))
        
        # Test with custom modulus
        B = Mod(modulus=2.0)
        def src(t): return t, 3*t
        def ref(t): return np.mod(t, 2.0), np.mod(3*t, 2.0)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))


class TestClip(unittest.TestCase):
    """
    Test the implementation of the 'Clip' block class
    """

    def test_embedding(self):
        """test algebraic components via embedding"""
        
        # Test with default limits (-1.0, 1.0)
        B = Clip()
        def src(t): return t - 5  # Values that will be clipped
        def ref(t): return np.clip(t - 5, -1.0, 1.0) 
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))
        
        # Test with custom limits
        B = Clip(min_val=-2.0, max_val=3.0)
        def src(t): return t - 1, 2*t - 5
        def ref(t): return np.clip(t - 1, -2.0, 3.0), np.clip(2*t - 5, -2.0, 3.0)
        E = Embedding(B, src, ref)
        
        for t in range(10): self.assertTrue(np.allclose(*E.check_MIMO(t)))



class TestMatrix(unittest.TestCase):
    """
    Test the implementation of the 'Matrix' block class
    """

    def test_init_default(self):
        """test default initialization with identity matrix"""
        B = Matrix()
        self.assertEqual(len(B.inputs), 1)
        self.assertEqual(len(B.outputs), 1)
        self.assertTrue(np.allclose(B.A, np.eye(1)))

    def test_init_custom_square(self):
        """test initialization with custom square matrix"""
        A = np.array([[1, 2], [3, 4]])
        B = Matrix(A=A)
        self.assertEqual(len(B.inputs), 2)
        self.assertEqual(len(B.outputs), 2)
        self.assertTrue(np.allclose(B.A, A))

    def test_init_custom_rectangular(self):
        """test initialization with non-square matrix"""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        B = Matrix(A=A)
        self.assertEqual(len(B.inputs), 3)
        self.assertEqual(len(B.outputs), 2)

    def test_init_invalid_dimension(self):
        """test that 1D array raises error"""
        with self.assertRaises(ValueError):
            Matrix(A=np.array([1, 2, 3]))

    def test_init_invalid_dimension_3d(self):
        """test that 3D array raises error"""
        with self.assertRaises(ValueError):
            Matrix(A=np.ones((2, 2, 2)))

    def test_embedding_identity(self):
        """test algebraic components with identity matrix"""
        B = Matrix(A=np.eye(2))

        def src(t): return t + 1, 2*t + 1
        def ref(t): return np.array([t + 1, 2*t + 1])
        E = Embedding(B, src, ref)

        for t in range(10):
            self.assertTrue(np.allclose(*E.check_MIMO(t)))

    def test_embedding_square_matrix(self):
        """test algebraic components with square matrix"""
        A = np.array([[2, 1], [0, 3]])
        B = Matrix(A=A)

        def src(t): return t + 1, t + 2
        def ref(t):
            u = np.array([t + 1, t + 2])
            return np.dot(A, u)
        E = Embedding(B, src, ref)

        for t in range(10):
            self.assertTrue(np.allclose(*E.check_MIMO(t)))

    def test_embedding_rectangular_matrix(self):
        """test algebraic components with non-square matrix"""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix: 3 inputs, 2 outputs
        B = Matrix(A=A)

        def src(t): return t, t + 1, t + 2
        def ref(t):
            u = np.array([t, t + 1, t + 2])
            return np.dot(A, u)
        E = Embedding(B, src, ref)

        for t in range(10):
            self.assertTrue(np.allclose(*E.check_MIMO(t)))

    def test_embedding_single_output(self):
        """test with row vector matrix (multiple inputs, single output)"""
        A = np.array([[1, 2, 3]])  # 1x3 matrix
        B = Matrix(A=A)

        def src(t): return t, 2*t, 3*t
        def ref(t):
            u = np.array([t, 2*t, 3*t])
            return np.dot(A, u)
        E = Embedding(B, src, ref)

        for t in range(10):
            self.assertTrue(np.allclose(*E.check_MIMO(t)))



# RUN TESTS LOCALLY ====================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)