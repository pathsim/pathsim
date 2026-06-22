########################################################################################
##
##                                  TESTS FOR
##                            'blocks.constraint.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.constraint import AlgebraicConstraint

from tests.pathsim.blocks._embedding import Embedding


# TESTS ================================================================================

class TestAlgebraicConstraint(unittest.TestCase):
    """
    Test the implementation of the 'AlgebraicConstraint' block class
    """

    def test_init(self):

        def func(x, u): return x**2 - u

        AC = AlgebraicConstraint(func, x0=1.0)

        #residual stored and callable
        self.assertEqual(AC.func(2.0, 4.0), 0.0)

        #initial guess as float array
        np.testing.assert_array_equal(AC.x0, np.array([1.0]))
        np.testing.assert_array_equal(AC._x, np.array([1.0]))

        #output pre-sized to unknown dimension
        self.assertEqual(len(AC.outputs), 1)

        #input validation
        for v in [2, 0.3, 1j, np.ones(3)]:
            with self.assertRaises(ValueError):
                AlgebraicConstraint(func=v)


    def test_algebraic_path(self):
        #purely algebraic block -> length 1 when active
        AC = AlgebraicConstraint(lambda x, u: x**2 - u, x0=1.0)
        self.assertEqual(len(AC), 1)
        AC.off()
        self.assertEqual(len(AC), 0)


    def test_scalar_constraint(self):
        #solve x**2 - u = 0 -> x = sqrt(u)
        AC = AlgebraicConstraint(lambda x, u: x**2 - u, x0=1.0)

        def src(t): return 2.0 + t          #positive input
        def ref(t): return np.sqrt(2.0 + t)

        E = Embedding(AC, src, ref)
        for t in range(5):
            out, exp = E.check_SISO(t)
            self.assertAlmostEqual(float(out), exp, places=8)


    def test_scalar_constraint_with_jac(self):
        #same problem with analytical jacobian d/dx (x**2 - u) = 2x
        AC = AlgebraicConstraint(
            func=lambda x, u: x**2 - u,
            x0=1.0,
            jac=lambda x, u: 2.0 * x
            )

        def src(t): return 3.0 + t
        def ref(t): return np.sqrt(3.0 + t)

        E = Embedding(AC, src, ref)
        for t in range(5):
            out, exp = E.check_SISO(t)
            self.assertAlmostEqual(float(out), exp, places=8)


    def test_vector_constraint(self):
        #reversible reaction equilibrium with mass conservation:
        #   k_f*xA - k_r*xB = 0,  xA + xB = u  ->  xA = u*k_r/(k_f+k_r)
        k_f, k_r = 2.0, 1.0

        def func(x, u):
            return np.array([k_f*x[0] - k_r*x[1], x[0] + x[1] - u[0]])

        AC = AlgebraicConstraint(func, x0=[0.5, 0.5])

        u = 6.0
        AC.inputs[0] = u
        AC.update(0.0)

        xA = u * k_r / (k_f + k_r)
        xB = u * k_f / (k_f + k_r)
        np.testing.assert_allclose(AC.outputs.to_array(), [xA, xB], atol=1e-8)


    def test_warmstart_and_reset(self):
        AC = AlgebraicConstraint(lambda x, u: x**2 - u, x0=1.0)

        #solve at u=4 -> x=2, warm-start advances
        AC.inputs[0] = 4.0
        AC.update(0.0)
        self.assertAlmostEqual(float(AC._x[0]), 2.0, places=8)

        #reset restores the initial guess and output
        AC.reset()
        np.testing.assert_array_equal(AC._x, AC.x0)
        np.testing.assert_array_equal(AC.outputs.to_array(), AC.x0)


    def test_info(self):
        #metadata introspection is available like for every block
        info = AlgebraicConstraint.info()
        self.assertEqual(info["type"], "AlgebraicConstraint")
        self.assertIn("func", info["parameters"])
        self.assertIn("x0", info["parameters"])
        self.assertIn("jac", info["parameters"])


    def test_simulation_steady_state(self):
        #drive the constraint with a constant source and integrate a bit:
        #the output must equal the implicit solution sqrt(u) throughout
        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        src = Constant(2.0)
        ac = AlgebraicConstraint(lambda x, u: x**2 - u, x0=1.0)
        sco = Scope()

        sim = Simulation(
            blocks=[src, ac, sco],
            connections=[Connection(src, ac), Connection(ac, sco)],
            log=False
            )
        sim.run(1.0)

        time, [data] = sco.read()
        np.testing.assert_allclose(data, np.sqrt(2.0), atol=1e-6)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
