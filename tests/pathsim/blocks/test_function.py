########################################################################################
##
##                                  TESTS FOR 
##                            'blocks.function.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.function import Function, DynamicalFunction

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestFunction(unittest.TestCase):
    """
    Test the implementation of the 'Function' block class
    """

    def test_init(self):

        def f(a): return a**2
        
        F = Function(func=f)

        #test if the function works
        self.assertEqual(F.func(1), f(1))
        self.assertEqual(F.func(2), f(2))
        self.assertEqual(F.func(3), f(3))

        #test input validation
        for v in [2, 0.3, 1j, np.ones(3)]:
            with self.assertRaises(ValueError): 
                F = Function(func=v)


    def test_embedding_siso(self):

        def f(a): return a**2
        
        F = Function(func=f)
        
        def src(t): return np.sin(t)
        def ref(t): return np.sin(t)**2
        
        E = Embedding(F, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        def src(t): return np.tanh(t)
        def ref(t): return np.tanh(t)**2

        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_embedding_miso(self):

        def f(a, b, c): return a**2 + b - c
        
        F = Function(func=f)

        def src(t): return np.sin(t), np.cos(t), np.tanh(t)
        def ref(t): return np.sin(t)**2 + np.cos(t) - np.tanh(t)
        
        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))



    def test_linearization_miso(self):
        """test linearization and delinearization"""

        def f(a, b, c): return a**2 + b - c
        
        F = Function(func=f)

        def src(t): return np.cos(t), t, 3.0
        def ref(t): return np.cos(t)**2 + t - 3.0 

        E = Embedding(F, src, ref)

        for t in range(10): 
            self.assertEqual(*E.check_MIMO(t))

        #linearize block
        F.linearize(t)

        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        F.delinearize()

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_update_siso(self):

        def f(a):
            return a**2
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], f(3))


    def test_update_miso(self):

        def f(a, b, c):
            return a**2 + b - c
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3
        F.inputs[1] = 2
        F.inputs[2] = 1

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], f(3, 2, 1))


    def test_update_simo(self):

        def f(a):
            return a**2, 2*a, 1
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], 9)
        self.assertEqual(F.outputs[1], 6)
        self.assertEqual(F.outputs[2], 1)


    def test_update_mimo(self):

        def f(a, b, c):
            return a**2-b, 3*c
        
        F = Function(func=f)

        #set block inputs
        F.inputs[0] = 3
        F.inputs[1] = 2
        F.inputs[2] = 1

        #update block
        F.update(None)

        #test if update was correct
        self.assertEqual(F.outputs[0], 7)
        self.assertEqual(F.outputs[1], 3)



class TestDynamicalFunction(unittest.TestCase):
    """
    Test the implementation of the 'DynamicalFunction' block class
    """

    def test_init(self):

        def f(a, t): 
            return a**2 + t
        
        F = DynamicalFunction(func=f)

        #test if the function works
        self.assertEqual(F.op_alg(None, 1, 2), f(1, 2))
        self.assertEqual(F.op_alg(None, 2, 3), f(2, 3))
        self.assertEqual(F.op_alg(None, 3, 4), f(3, 4))

        #test input validation
        for v in [2, 0.3, 1j, np.ones(3)]:
            with self.assertRaises(ValueError): 
                F = Function(func=v)


    def test_embedding_siso(self):

        def f(a, t): 
            return a**2 + t
        
        F = DynamicalFunction(func=f)
        
        def src(t): return np.sin(t)
        def ref(t): return np.sin(t)**2 + t
        
        E = Embedding(F, src, ref)
        
        for t in range(10): self.assertEqual(*E.check_SISO(t))

        def src(t): return np.tanh(t)
        def ref(t): return np.tanh(t)**2 + t

        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_SISO(t))


    def test_embedding_miso(self):

        def f(u, t):
            a, b, c = u
            return (a**2 + b - c) * t

        F = DynamicalFunction(func=f)

        def src(t): return np.sin(t), np.cos(t), np.tanh(t)
        def ref(t): return (np.sin(t)**2 + np.cos(t) - np.tanh(t))*t

        E = Embedding(F, src, ref)

        for t in range(10): self.assertEqual(*E.check_MIMO(t))


    def test_linearization_miso(self):
        """test linearization and delinearization

        regression test for a bug where 'Block.linearize' crashed with a
        'TypeError' for stateless blocks whose 'op_alg' is a
        'DynamicOperator' (as opposed to a plain 'Operator'), because it
        called 'op_alg.linearize(u)' with a single argument regardless of
        which operator type was involved
        """

        def f(u, t):
            a, b, c = u
            return (a**2 + b - c) * t

        F = DynamicalFunction(func=f)

        def src(t): return np.cos(t), t, 3.0
        def ref(t): return (np.cos(t)**2 + t - 3.0) * t

        E = Embedding(F, src, ref)

        for t in range(10):
            self.assertEqual(*E.check_MIMO(t))

        #linearize block
        F.linearize(t)

        a, b = E.check_MIMO(t)
        self.assertAlmostEqual(np.linalg.norm(a-b), 0, 8)

        #delinearize
        F.delinearize()

        for t in range(10):
            self.assertEqual(*E.check_MIMO(t))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)