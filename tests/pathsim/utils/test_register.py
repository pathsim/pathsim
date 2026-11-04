########################################################################################
##
##                                  TESTS FOR 
##                             'utils.register.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.register import Register


# TESTS ================================================================================

class TestRegister(unittest.TestCase):
    """
    test the 'Register' class
    """

    def test_init(self):
        #default
        R = Register()
        self.assertTrue(isinstance(R._data, np.ndarray))
        self.assertEqual(len(R._data), 1)
        self.assertEqual(R[0], 0.0)

        #specific
        R = Register(size=20)
        self.assertTrue(isinstance(R._data, np.ndarray))
        self.assertEqual(len(R._data), 20)
        self.assertEqual(R[0], 0.0)
        self.assertEqual(R[19], 0.0)

        #accessing a key beyond initial size returns default
        self.assertEqual(R[20], 0.0)

        #length doesn't increase when only accessing
        self.assertEqual(len(R), 20)


    def test_len(self):
        #default
        R = Register()
        self.assertEqual(len(R), 1)

        #specific
        R = Register(size=5)
        self.assertEqual(len(R), 5)

        # After adding items (dense array includes all indices)
        R[5] = 10.0
        R[10] = 20.0
        self.assertEqual(len(R), 11)  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10


    def test_iter(self):
        #specific
        R = Register(3)
        R[0] = 1.0
        R[1] = 2.0
        R[2] = 3.0

        #add out of order (dense array fills intermediate indices with 0.0)
        R[5] = 5.0
        R[4] = 4.0
        expected = [1.0, 2.0, 3.0, 0.0, 4.0, 5.0]

        #iteration includes all elements
        np.testing.assert_array_equal([v for v in R], expected)
        self.assertEqual(len([v for v in R]), 6)


    def test_reset(self):

        R = Register(3)

        R[0] = 1.0
        R[1] = -2.5
        R[2] = 100.0
        R[5] = 99.0

        self.assertEqual(len(R), 6)  # dense array 0-5

        R.reset()

        #length remains same
        self.assertEqual(len(R), 6)

        self.assertEqual(R[0], 0.0)
        self.assertEqual(R[1], 0.0)
        self.assertEqual(R[2], 0.0)

        #reset dynamic key too
        self.assertEqual(R[5], 0.0)

        #default value for non-existent key after reset
        self.assertEqual(R[10], 0.0)


    def test_to_array(self):
        R = Register(3)
        R[0] = 1.1
        R[1] = 2.2
        R[2] = 3.3

        np.testing.assert_array_equal(
            R.to_array(),
            np.array([1.1, 2.2, 3.3])
            )

        #test with dynamic keys added out of order (includes zero at index 3)
        R[5] = 5.5
        R[4] = 4.4

        np.testing.assert_array_equal(
            R.to_array(),
            np.array([1.1, 2.2, 3.3, 0.0, 4.4, 5.5])
            )

        #test empty initialized (should be size 1 default)
        R = Register()

        np.testing.assert_array_equal(
            R.to_array(), 
            np.array([0.0])
            )


    def test_update_from_array(self):

        #test with array
        R = Register(3)
        arr = np.array([10.1, 20.2, 30.3])
        R.update_from_array(arr)
        self.assertEqual(R[0], 10.1)
        self.assertEqual(R[1], 20.2)
        self.assertEqual(R[2], 30.3)

        #test that size increased implicitly if array was larger
        arr_large = np.array([1, 2, 3, 4, 5])
        R.update_from_array(arr_large)
        self.assertEqual(len(R), 5)
        self.assertEqual(R[4], 5.0)

        #test with scalar
        R = Register(1)
        R.update_from_array(99.9)
        self.assertEqual(R[0], 99.9)
        self.assertEqual(len(R), 1)

        #test scalar update on multi-port register (only updates port 0)
        R = Register(3)
        R[1] = 1.0 
        R.update_from_array(5.5)
        self.assertEqual(R[0], 5.5)
        self.assertEqual(R[1], 1.0) 


    def test_setitem(self):

        R = Register(1)

        #set initial value
        R[0] = 10.0
        self.assertEqual(R._data[0], 10.0)

        #add new key dynamically (resizes array)
        R[5] = 50.5
        self.assertEqual(R._data[5], 50.5)

        #dense array includes all indices 0-5
        self.assertEqual(len(R), 6)

        #overwrite
        R[0] = -1.0
        self.assertEqual(R._data[0], -1.0)


    def test_getitem(self):

        R = Register(2)
        
        R[0] = 1.1
        R[1] = 2.2

        #get existing keys
        self.assertEqual(R[0], 1.1)
        self.assertEqual(R[1], 2.2)
        
        #get non-existing key (should return default float 0.0)
        self.assertEqual(R[5], 0.0)
        
        #check length has not changed just by getting default
        self.assertEqual(len(R), 2)



    def test_mapping(self):

        R = Register(mapping={"a": 0, "b": 2})

        #register is sized to hold all aliases
        self.assertEqual(len(R), 3)

        #set and get by alias
        R["b"] = 3.3
        self.assertEqual(R["b"], 3.3)
        self.assertEqual(R[2], 3.3)

        #unknown alias returns default float 0.0
        self.assertEqual(R["c"], 0.0)

        #explicit size larger than the aliases
        self.assertEqual(len(Register(5, mapping={"a": 0})), 5)

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)