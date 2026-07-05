import unittest
import numpy as np

from pathsim.blocks.table import LUT, LUT1D

from tests.pathsim.blocks._embedding import Embedding


class TestLUT(unittest.TestCase):
    """
    Test the implementation of the 'LUT' (Look Up Table) block class
    """

    def test_init(self):
        """Test initialization of LUT block"""
        # 2D lookup table
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([0, 1, 1, 2])  # z = x + y
        
        lut = LUT(points, values)
        
        # Test that function was properly initialized
        self.assertTrue(callable(lut.func))
        self.assertIsNotNone(lut.inter)


    def test_update_2d_mimo(self):
        """Test update method for 2D MIMO case"""
        # 2D input, 2D output lookup table
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([[0, 0], [1, 2], [1, 2], [2, 4]])  # [x+y, 2*(x+y)]
        
        lut = LUT(points, values)
        
        # Set block inputs
        lut.inputs[0] = 1
        lut.inputs[1] = 1
        
        # Update block
        lut.update(None)
        
        # Test if update was correct
        self.assertAlmostEqual(lut.outputs[0], 2, places=6)
        self.assertAlmostEqual(lut.outputs[1], 4, places=6)


class TestLUT1DEmbedding(unittest.TestCase):
    """
    Test the implementation of the 'LUT1D' (1D Look Up Table) block class
    """

    def test_init(self):
        """Test initialization of LUT1D block"""
        # Simple 1D lookup table
        points = np.array([0, 1, 2, 3])
        values = np.array([0, 1, 4, 9])  # y = x^2

        lut = LUT1D(points, values)

        # Test that function was properly initialized
        self.assertTrue(callable(lut.func))
        self.assertIsNotNone(lut.inter)

        # Test input validation
        with self.assertRaises(ValueError):
            LUT1D(points, "invalid")  # Invalid values type

        with self.assertRaises(ValueError):
            LUT1D("invalid", values)  # Invalid points type

    def test_embedding_siso(self):
        """Test 1D SISO embedding"""
        # Create lookup table for y = 2*x
        points = np.array([0, 1, 2, 3, 4])
        values = np.array([0, 2, 4, 6, 8])
        
        lut = LUT1D(points, values)
        
        def src(t): 
            return np.array([t % 4])  # Keep within domain
        def ref(t): 
            return 2 * (t % 4)  # Expected output
        
        E = Embedding(lut, src, ref)
        
        # Test for several time points
        for t in [0, 1, 2, 3]:
            y, r = E.check_SISO(t)
            self.assertAlmostEqual(y, r, places=6)

    def test_embedding_simo(self):
        """Test 1D SIMO embedding"""
        # 1D input, 2D output lookup table
        points = np.array([0, 1, 2])
        values = np.array([[0, 0], [1, 2], [4, 4]])  # [x^2, 2*x]
        
        lut = LUT1D(points, values)
        
        def src(t): 
            return np.array([t % 2])
        def ref(t): 
            x = t % 2
            return np.array([x**2, 2*x])
        
        E = Embedding(lut, src, ref)
        
        # Test for several time points
        for t in [0, 1]:
            y, r = E.check_MIMO(t)
            np.testing.assert_array_almost_equal(y, r, decimal=6)

    def test_linearization_siso(self):
        """Test linearization and delinearization for SISO case"""
        # Nonlinear function approximated by lookup table
        points = np.array([0, 1, 2, 3])
        values = points**2  # y = x^2
        
        lut = LUT1D(points, values)
        
        def src(t):
            return np.array([1.5])
        def ref(t):
            return 2.5  # linear interpolation between (1, 1) and (2, 4)
        
        E = Embedding(lut, src, ref)
        
        # Test original nonlinear behavior
        y, r = E.check_SISO(0)
        self.assertAlmostEqual(y, r, places=6)
        
        # Linearize block at t=0
        lut.linearize(0)
        
        # At linearization point, should still match
        y, r = E.check_SISO(0)
        self.assertAlmostEqual(y, r, places=6)
        
        # Delinearize
        lut.delinearize()
        
        # Should return to original nonlinear behavior
        y, r = E.check_SISO(0)
        self.assertAlmostEqual(y, r, places=6)

    @unittest.skip("'pathsim.optim.value.Value' does not exist in the current codebase")
    def test_sensitivity(self):
        """Test compatibility with AD framework"""
        from pathsim.optim.value import Value
        
        # Create Value objects for sensitivity analysis
        x = Value(1.5, name='x')
        
        # Simple lookup table
        points = np.array([0, 1, 2, 3])
        values = points**2  # y = x^2
        
        lut = LUT1D(points, values)
        
        def src(t): 
            return np.array([x])
        def ref(t): 
            return x**2
        
        E = Embedding(lut, src, ref)
        
        y, r = E.check_SISO(0)
        self.assertAlmostEqual(y.val, r.val, places=6)
        
        # Check derivative (should be approximately 2*x = 3 at x=1.5)
        self.assertAlmostEqual(Value.der(y, x), 3.0, places=1)

    def test_update_siso(self):
        """Test update method for SISO case"""
        # Simple lookup table
        points = np.array([0, 1, 2])
        values = np.array([0, 10, 20])
        
        lut = LUT1D(points, values)
        
        # Set block inputs
        lut.inputs[0] = 1
        
        # Update block
        lut.update(None)
        
        # Test if update was correct
        self.assertEqual(lut.outputs[0], 10)

    def test_update_simo(self):
        """Test update method for SIMO case"""
        # 1D input, 2D output lookup table
        points = np.array([0, 1, 2])
        values = np.array([[0, 0], [1, 10], [4, 40]])  # [x^2, x*10]
        
        lut = LUT1D(points, values)
        
        # Set block inputs
        lut.inputs[0] = 1
        
        # Update block
        lut.update(None)
        
        # Test if update was correct
        self.assertAlmostEqual(lut.outputs[0], 1, places=6)
        self.assertAlmostEqual(lut.outputs[1], 10, places=6)

    def test_extrapolation_behavior(self):
        """Test extrapolation behavior"""
        points = np.array([1, 2, 3])
        values = np.array([1, 4, 9])
        
        lut = LUT1D(points, values)
        
        # Test extrapolation outside domain
        lut.inputs[0] = 0  # Below domain
        lut.update(None)
        self.assertFalse(np.isnan(lut.outputs[0]))  # Should extrapolate, not NaN
        
        lut.inputs[0] = 4  # Above domain
        lut.update(None)
        self.assertFalse(np.isnan(lut.outputs[0]))  # Should extrapolate, not NaN



class TestLUT1D(unittest.TestCase):
    """
    Test the implementation of the 'LUT1D' (1D Look Up Table) block class
    """

    def test_init_1d_single_output(self):
        """Test initialization with 1D data and single output"""
        # Simple 1D lookup table
        points = np.array([0, 1, 2, 3])
        values = np.array([0, 1, 4, 9])  # y = x^2
        
        lut = LUT1D(points, values)
        
        # Test that function was properly initialized
        self.assertTrue(callable(lut.func))
        self.assertIsNotNone(lut.inter)
        self.assertIsNotNone(lut.op_alg)

    def test_init_1d_multiple_outputs(self):
        """Test initialization with 1D data and multiple outputs"""
        # 1D input, 2D output lookup table
        points = np.array([0, 1, 2])
        values = np.array([[0, 0], [1, 10], [4, 40]])  # [x^2, x*10]
        
        lut = LUT1D(points, values)
        
        # Test that function was properly initialized
        self.assertTrue(callable(lut.func))
        self.assertIsNotNone(lut.inter)
        self.assertIsNotNone(lut.op_alg)

    def test_interpolation_1d_multiple_outputs(self):
        """Test 1D interpolation functionality with multiple outputs"""
        # 1D input, 2D output lookup table
        points = np.array([0, 1, 2])
        values = np.array([[0, 0], [1, 10], [4, 40]])  # [x^2, x*10]
        
        lut = LUT1D(points, values)
        
        # Test exact points
        lut.inputs[0] = 1
        lut.update(0)
        self.assertAlmostEqual(lut.outputs[0], 1, places=6)
        self.assertAlmostEqual(lut.outputs[1], 10, places=6)
        
        # Test interpolation
        lut.inputs[0] = 0.5  # Halfway between 0 and 1
        lut.update(0)
        self.assertAlmostEqual(lut.outputs[0], 0.5, places=6)  # (0+1)/2
        self.assertAlmostEqual(lut.outputs[1], 5, places=6)    # (0+10)/2

    def test_update_functionality_multiple_outputs(self):
        """Test the update method directly with multiple outputs"""
        # 1D input, 2D output lookup table
        points = np.array([0, 1, 2])
        values = np.array([[0, 0], [1, 10], [4, 40]])  # [x^2, x*10]
        
        lut = LUT1D(points, values)
        
        # Test update with exact points
        lut.inputs[0] = 1
        lut.update(0)
        self.assertAlmostEqual(lut.outputs[0], 1, places=6)
        self.assertAlmostEqual(lut.outputs[1], 10, places=6)
        
        # Test interpolation
        lut.inputs[0] = 0.5
        lut.update(0)
        self.assertAlmostEqual(lut.outputs[0], 0.5, places=6)
        self.assertAlmostEqual(lut.outputs[1], 5, places=6)

    def test_inheritance_from_function(self):
        """Test that LUT1D properly inherits from Function block"""
        points = np.array([0, 1])
        values = np.array([0, 1])
        
        lut = LUT1D(points, values)
        
        # Should have Function block attributes
        self.assertTrue(hasattr(lut, 'func'))
        self.assertTrue(hasattr(lut, 'op_alg'))
        self.assertTrue(hasattr(lut, 'inputs'))
        self.assertTrue(hasattr(lut, 'outputs'))
        
        # Should be callable like Function
        self.assertTrue(callable(lut.func))
