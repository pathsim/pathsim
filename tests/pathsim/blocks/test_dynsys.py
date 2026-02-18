########################################################################################
##
##                                  TESTS FOR 
##                             'blocks.dynsys.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.dynsys import DynamicalSystem

#base solver for testing
from pathsim.solvers._solver import Solver
from pathsim.solvers.euler import EUF, EUB

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestDynamicalSystem(unittest.TestCase):
    """
    Test the implementation of the 'DynamicalSystem' block class
    """

    def test_init(self):
        """Test default and custom initialization"""
        
        #test default initialization
        D = DynamicalSystem()
        self.assertEqual(D.initial_value, 0.0)
        self.assertEqual(D.engine, None)
        self.assertEqual(len(D.inputs), 1)
        self.assertEqual(len(D.outputs), 1)

        #test custom initialization (scalar)
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -2*x + u,
            func_alg=lambda x, u, t: 3*x,
            initial_value=1.5
        )
        self.assertEqual(D.initial_value, 1.5)
        self.assertEqual(D.func_dyn(1, 2, 0), 0)  # -2*1 + 2 = 0
        self.assertEqual(D.func_alg(2, 0, 0), 6)  # 3*2 = 6

        #test vector initialization
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: x + u,
            initial_value=np.array([1.0, 2.0])
        )
        self.assertTrue(np.all(D.initial_value == np.array([1.0, 2.0])))


    def test_len_passthrough(self):
        """Test algebraic passthrough detection"""
        
        #no passthrough - output doesn't depend on input
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: x,
            initial_value=1.0
        )
        D.set_solver(Solver, None)
        self.assertEqual(len(D), 0.0)

        #has passthrough - output depends on input
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x + u,
            func_alg=lambda x, u, t: x + u,
            initial_value=1.0
        )
        D.set_solver(Solver, None)
        self.assertEqual(len(D), 1)

        #vector case with partial passthrough
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: np.array([x[0], u[0]]),
            initial_value=np.array([1.0, 2.0])
        )
        D.set_solver(Solver, None)
        self.assertEqual(len(D), 1)


    def test_set_solver(self):
        """Test solver initialization and parameter updates"""
        
        D = DynamicalSystem(initial_value=1.0)

        #test that no solver is initialized
        self.assertEqual(D.engine, None)

        D.set_solver(Solver, None, tolerance_lte_rel=1e-4, tolerance_lte_abs=1e-6)

        #test that solver is now available
        self.assertTrue(isinstance(D.engine, Solver))

        #test that solver parameters have been set
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-4)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-6)
        self.assertEqual(D.engine.initial_value, 1.0)

        #change solver parameters
        D.set_solver(Solver, None, tolerance_lte_rel=1e-2, tolerance_lte_abs=1e-3)

        #test that solver tolerance is changed
        self.assertEqual(D.engine.tolerance_lte_rel, 1e-2)
        self.assertEqual(D.engine.tolerance_lte_abs, 1e-3)


    def test_update(self):
        """Test algebraic output equation evaluation"""
        
        #simple gain system
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: 0*x,  # static state
            func_alg=lambda x, u, t: 2*x + u,
            initial_value=1.5
        )
        D.set_solver(Solver, None)

        #test initial output
        self.assertEqual(D.outputs[0], 0.0)
        
        D.inputs[0] = 3.0
        D.update(0)

        #test if output is calculated correctly: 2*1.5 + 3.0 = 6.0
        self.assertAlmostEqual(D.outputs[0], 6.0, 8)


    def test_jacobian_usage(self):
        """Test that provided jacobian is used in solve"""
        
        #linear system with analytical jacobian
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -2*x + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2.0
        )
        D.set_solver(Solver, None)

        #verify operators were created
        self.assertIsNotNone(D.op_dyn)
        self.assertIsNotNone(D.op_alg)

        #test jacobian evaluation
        x, u, t = 1.0, 0.0, 0.0
        jac = D.op_dyn.jac_x(x, u, t)
        self.assertEqual(jac, -2.0)


    def test_state_space_equivalence(self):
        """Test that DynamicalSystem can replicate StateSpace behavior"""
        
        #simple state space: dx/dt = Ax + Bu, y = Cx + Du
        A, B, C, D_mat = -2.0, 1.0, 1.5, 0.5
        
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: A*x + B*u,
            func_alg=lambda x, u, t: C*x + D_mat*u,
            initial_value=1.0
        )
        D.set_solver(Solver, None)

        #apply input and check output matches expected
        D.inputs[0] = 2.0
        D.update(0)
        
        expected_output = C * D.engine.get() + D_mat * 2.0
        self.assertAlmostEqual(D.outputs[0], expected_output, 8)


    def test_time_varying_system(self):
        """Test system with explicit time dependence"""
        
        #time-varying gain: dx/dt = -x, y = sin(t)*x
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: np.sin(t) * x,
            initial_value=1.0
        )
        D.set_solver(Solver, None)

        #check output at different times
        D.update(0)
        output_t0 = D.outputs[0]
        
        D.update(np.pi/2)  # sin(pi/2) = 1
        output_tpi = D.outputs[0]
        
        #output should scale with sin(t)
        self.assertNotEqual(output_t0, output_tpi)


    def test_nonlinear_dynamics(self):
        """Test nonlinear system (Van der Pol oscillator simplified)"""

        #dx/dt = x - x^3
        D = DynamicalSystem(
            func_dyn=lambda x, u, t: x - x**3,
            func_alg=lambda x, u, t: x,
            initial_value=0.5
        )
        D.set_solver(Solver, None)

        #verify dynamics: at x=0.5, dx/dt = 0.5 - 0.125 = 0.375
        result = D.func_dyn(0.5, 0, 0)
        self.assertAlmostEqual(result, 0.375, 8)


    def test_solve(self):
        """Test implicit solve path"""

        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -2*x + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2.0
        )
        D.set_solver(EUB, None)

        #buffer state before solve
        D.engine.buffer(0.01)

        D.inputs[0] = 0.5
        err = D.solve(0, 0.01)

        self.assertIsInstance(err, (int, float, np.floating))


    def test_step(self):
        """Test explicit step path"""

        D = DynamicalSystem(
            func_dyn=lambda x, u, t: -x,
            func_alg=lambda x, u, t: x,
            initial_value=1.0
        )
        D.set_solver(EUF, None)

        #buffer state before step
        D.engine.buffer(0.01)

        D.inputs[0] = 0.0
        success, error, scale = D.step(0, 0.01)

        self.assertTrue(success)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)