########################################################################################
##
##                                  TESTS FOR
##                               'blocks.dae.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.dae import SemiExplicitDAE

from pathsim.solvers.esdirk43 import ESDIRK43


# TESTS ================================================================================

class TestSemiExplicitDAE(unittest.TestCase):
    """
    Test the implementation of the 'SemiExplicitDAE' block class.

    Reference problem (scalar, autonomous):

        x' = -x + z,   0 = z - x**2      =>   x' = x**2 - x

    with x(0) = 0.5 this has the closed-form solution

        x(t) = 1 / (1 + exp(t)),   z(t) = x(t)**2
    """

    def _make_block(self):
        f_dyn = lambda x, z, u, t: -x + z
        f_alg = lambda x, z, u, t: z - x**2
        return SemiExplicitDAE(f_dyn, f_alg, initial_value=0.5, z0=0.25)

    @staticmethod
    def _exact(t):
        x = 1.0 / (1.0 + np.exp(t))
        return x, x**2


    def test_init(self):
        dae = self._make_block()

        #differential initial condition drives the engine
        np.testing.assert_array_equal(dae.initial_value, np.array([0.5]))

        #algebraic warm-start
        np.testing.assert_array_equal(dae.z0, np.array([0.25]))
        np.testing.assert_array_equal(dae._z, np.array([0.25]))

        #output pre-sized to the stacked state [x, z]
        self.assertEqual(len(dae.outputs), 2)


    def test_z_elimination(self):
        #constraint z - x**2 = 0 -> z = x**2
        dae = self._make_block()
        z = dae._solve_z(np.array([0.3]), np.array([0.0]), 0.0)
        self.assertAlmostEqual(float(z[0]), 0.09, places=8)


    def test_reduced_rhs(self):
        #reduced right hand side x' = -x + z = x**2 - x
        dae = self._make_block()
        dx = dae._rhs(np.array([0.3]), np.array([0.0]), 0.0)
        self.assertAlmostEqual(float(dx[0]), 0.3**2 - 0.3, places=8)


    def test_algebraic_path(self):
        dae = self._make_block()
        self.assertEqual(len(dae), 1)
        dae.off()
        self.assertEqual(len(dae), 0)


    def test_jac_z_matches_numerical(self):
        #analytical jac_z must give the same elimination as the FD fallback
        f_dyn = lambda x, z, u, t: -x + z
        f_alg = lambda x, z, u, t: z - x**2
        jac_z = lambda x, z, u, t: np.array([[1.0]])

        dae_num = SemiExplicitDAE(f_dyn, f_alg, 0.5, 0.25)
        dae_ana = SemiExplicitDAE(f_dyn, f_alg, 0.5, 0.25, jac_z=jac_z)

        x, u = np.array([0.4]), np.array([0.0])
        np.testing.assert_allclose(
            dae_num._solve_z(x, u, 0.0),
            dae_ana._solve_z(x, u, 0.0),
            atol=1e-10
            )


    def test_reset(self):
        dae = self._make_block()
        dae.set_solver(ESDIRK43, None)
        dae._z = np.array([0.99])
        dae.reset()
        np.testing.assert_array_equal(dae._z, dae.z0)


    def test_info(self):
        info = SemiExplicitDAE.info()
        self.assertEqual(info["type"], "SemiExplicitDAE")
        for p in ("func_dyn", "func_alg", "initial_value", "z0", "jac_z"):
            self.assertIn(p, info["parameters"])


    def test_simulation_matches_analytic(self):
        #integrate the DAE with an implicit solver and compare to the exact
        #solution, exercising the reduced right hand side and its jacobian
        from pathsim import Simulation, Connection
        from pathsim.blocks import Scope

        dae = self._make_block()
        sco = Scope()

        sim = Simulation(
            blocks=[dae, sco],
            connections=[Connection(dae[0], sco[0])],
            dt=0.01,
            Solver=ESDIRK43,
            log=False
            )
        sim.run(2.0)

        x_exact, z_exact = self._exact(2.0)
        self.assertAlmostEqual(float(dae.engine.state[0]), x_exact, places=4)
        self.assertAlmostEqual(float(dae._z[0]), z_exact, places=4)


    def test_simulation_with_input(self):
        #constraint pulls z to the input, the state relaxes to x -> u:
        #   x' = -x + z,   0 = z - u   =>   x' = -x + u
        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        f_dyn = lambda x, z, u, t: -x + z
        f_alg = lambda x, z, u, t: z - u

        dae = SemiExplicitDAE(f_dyn, f_alg, initial_value=0.0, z0=0.0)
        src = Constant(3.0)
        sco = Scope()

        sim = Simulation(
            blocks=[src, dae, sco],
            connections=[Connection(src, dae), Connection(dae[0], sco[0])],
            dt=0.01,
            Solver=ESDIRK43,
            log=False
            )
        sim.run(20.0)

        #relaxed to the steady state x -> u = 3
        self.assertAlmostEqual(float(dae.engine.state[0]), 3.0, places=4)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
