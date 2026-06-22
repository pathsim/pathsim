########################################################################################
##
##                                  TESTS FOR
##                               'blocks.dae.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.dae import SemiExplicitDAE, MassMatrixDAE, FullyImplicitDAE

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



class TestMassMatrixDAE(unittest.TestCase):
    """
    Test the implementation of the 'MassMatrixDAE' block class.
    """

    def test_init_nonsingular(self):
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        dae = MassMatrixDAE(lambda x, u, t: -x, M, initial_value=[1.0, 1.0])
        np.testing.assert_array_equal(dae._d, np.array([0, 1]))
        self.assertEqual(dae._a.size, 0)
        np.testing.assert_array_equal(dae.initial_value, np.array([1.0, 1.0]))
        self.assertEqual(len(dae.outputs), 2)

    def test_init_singular_partition(self):
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        func = lambda x, u, t: np.array([-x[0] + x[1], x[0] + x[1] - u[0]])
        dae = MassMatrixDAE(func, M, initial_value=[0.0, 0.0])
        np.testing.assert_array_equal(dae._d, np.array([0]))
        np.testing.assert_array_equal(dae._a, np.array([1]))
        np.testing.assert_array_equal(dae.initial_value, np.array([0.0]))

    def test_validation(self):
        func = lambda x, u, t: -x
        with self.assertRaises(ValueError):
            MassMatrixDAE(func, np.ones((2, 3)), initial_value=[0.0, 0.0])
        with self.assertRaises(ValueError):
            MassMatrixDAE(func, np.eye(2), initial_value=[0.0])
        M_bad = np.array([[1.0, 1.0], [0.0, 0.0]])
        with self.assertRaises(ValueError):
            MassMatrixDAE(func, M_bad, initial_value=[0.0, 0.0])

    def test_passthrough(self):
        dae_ns = MassMatrixDAE(lambda x, u, t: -x, np.eye(2), [1.0, 1.0])
        self.assertEqual(len(dae_ns), 0)
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        func = lambda x, u, t: np.array([-x[0] + x[1], x[0] + x[1] - u[0]])
        dae_s = MassMatrixDAE(func, M, [0.0, 0.0])
        self.assertEqual(len(dae_s), 1)

    def test_reset(self):
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        func = lambda x, u, t: np.array([-x[0] + x[1], x[0] + x[1] - u[0]])
        dae = MassMatrixDAE(func, M, [0.0, 0.0])
        dae.set_solver(ESDIRK43, None)
        dae._xa = np.array([5.0])
        dae.reset()
        np.testing.assert_array_equal(dae._xa, dae._x0[dae._a])

    def test_info(self):
        info = MassMatrixDAE.info()
        self.assertEqual(info["type"], "MassMatrixDAE")
        for p in ("func", "mass", "initial_value", "jac"):
            self.assertIn(p, info["parameters"])

    def test_simulation_nonsingular(self):
        #2 x0' = -x0,  x1' = -x1  ->  x0 = exp(-t/2),  x1 = exp(-t)
        from pathsim import Simulation, Connection
        from pathsim.blocks import Scope
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        dae = MassMatrixDAE(lambda x, u, t: -x, M, initial_value=[1.0, 1.0])
        sco = Scope()
        sim = Simulation(
            blocks=[dae, sco],
            connections=[Connection(dae[0], sco[0]), Connection(dae[1], sco[1])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.run(1.0)
        x = dae.engine.state
        self.assertAlmostEqual(float(x[0]), np.exp(-0.5), places=4)
        self.assertAlmostEqual(float(x[1]), np.exp(-1.0), places=4)

    def test_simulation_nonsingular_with_jac(self):
        #analytical jacobian path (nonsingular -> M^-1 df/dx)
        from pathsim import Simulation, Connection
        from pathsim.blocks import Scope
        M = np.array([[2.0, 0.0], [0.0, 1.0]])
        func = lambda x, u, t: -x
        jac = lambda x, u, t: -np.eye(2)
        dae = MassMatrixDAE(func, M, initial_value=[1.0, 1.0], jac=jac)
        sco = Scope()
        sim = Simulation(
            blocks=[dae, sco],
            connections=[Connection(dae[0], sco[0])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.run(1.0)
        self.assertAlmostEqual(float(dae.engine.state[0]), np.exp(-0.5), places=4)

    def test_simulation_singular_with_input(self):
        #x0' = -x0 + x1,  0 = x0 + x1 - u   =>   steady state x -> u/2
        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope
        M = np.array([[1.0, 0.0], [0.0, 0.0]])
        func = lambda x, u, t: np.array([-x[0] + x[1], x[0] + x[1] - u[0]])
        dae = MassMatrixDAE(func, M, initial_value=[0.0, 0.0])
        src = Constant(2.0)
        sco = Scope()
        sim = Simulation(
            blocks=[src, dae, sco],
            connections=[Connection(src, dae), Connection(dae[0], sco[0])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.run(20.0)
        np.testing.assert_allclose(dae.outputs.to_array(), [1.0, 1.0], atol=1e-3)


class TestFullyImplicitDAE(unittest.TestCase):
    """
    Test the implementation of the 'FullyImplicitDAE' block class.
    """

    def test_init(self):
        dae = FullyImplicitDAE(lambda x, xdot, u, t: xdot + x, initial_value=2.0)
        np.testing.assert_array_equal(dae.initial_value, np.array([2.0]))
        np.testing.assert_array_equal(dae._xdot, np.array([0.0]))
        self.assertEqual(len(dae.outputs), 1)

    def test_xdot_recovery(self):
        #F = xdot + x = 0 -> xdot = -x
        dae = FullyImplicitDAE(lambda x, xdot, u, t: xdot + x, initial_value=2.0)
        xdot = dae._solve_xdot(np.array([2.0]), np.array([0.0]), 0.0)
        self.assertAlmostEqual(float(xdot[0]), -2.0, places=8)

    def test_passthrough(self):
        dae = FullyImplicitDAE(lambda x, xdot, u, t: xdot + x, initial_value=1.0)
        self.assertEqual(len(dae), 0)

    def test_reduced_jac_analytic_matches_numerical(self):
        #F = xdot + x  ->  dF/dx = I, dF/dxdot = I  ->  reduced J = -I
        func = lambda x, xdot, u, t: xdot + x
        jac_x = lambda x, xdot, u, t: np.eye(len(x))
        jac_xdot = lambda x, xdot, u, t: np.eye(len(x))

        dae_num = FullyImplicitDAE(func, [1.0, 2.0])
        dae_ana = FullyImplicitDAE(func, [1.0, 2.0], jac_x=jac_x, jac_xdot=jac_xdot)

        x, u = np.array([1.0, 2.0]), np.array([0.0])
        #analytic reduced jacobian and the dynamic-operator (numerical) jacobian
        J_ana = dae_ana._reduced_jac(x, u, 0.0)
        J_num = dae_num.op_dyn.jac_x(x, u, 0.0)
        np.testing.assert_allclose(J_ana, -np.eye(2), atol=1e-12)
        np.testing.assert_allclose(J_num, -np.eye(2), atol=1e-6)

    def test_reset(self):
        dae = FullyImplicitDAE(lambda x, xdot, u, t: xdot + x, initial_value=2.0)
        dae.set_solver(ESDIRK43, None)
        dae._xdot = np.array([9.0])
        dae.reset()
        np.testing.assert_array_equal(dae._xdot, np.zeros(1))

    def test_info(self):
        info = FullyImplicitDAE.info()
        self.assertEqual(info["type"], "FullyImplicitDAE")
        for p in ("func", "initial_value", "jac_x", "jac_xdot"):
            self.assertIn(p, info["parameters"])

    def test_simulation_autonomous(self):
        #F = xdot + x = 0  ->  x' = -x  ->  x = exp(-t)
        from pathsim import Simulation, Connection
        from pathsim.blocks import Scope
        dae = FullyImplicitDAE(lambda x, xdot, u, t: xdot + x, initial_value=1.0)
        sco = Scope()
        sim = Simulation(
            blocks=[dae, sco], connections=[Connection(dae[0], sco[0])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.run(2.0)
        self.assertAlmostEqual(float(dae.engine.state[0]), np.exp(-2.0), places=4)

    def test_simulation_oscillator(self):
        #implicit harmonic oscillator, x0 = [1, 0] -> [cos t, -sin t]
        from pathsim import Simulation, Connection
        from pathsim.blocks import Scope
        def func(x, xdot, u, t):
            return np.array([xdot[0] - x[1], xdot[1] + x[0]])
        dae = FullyImplicitDAE(func, initial_value=[1.0, 0.0])
        sco = Scope()
        sim = Simulation(
            blocks=[dae, sco],
            connections=[Connection(dae[0], sco[0]), Connection(dae[1], sco[1])],
            dt=0.005, Solver=ESDIRK43, log=False
            )
        sim.run(1.0)
        x = dae.engine.state
        self.assertAlmostEqual(float(x[0]), np.cos(1.0), places=3)
        self.assertAlmostEqual(float(x[1]), -np.sin(1.0), places=3)

    def test_simulation_with_input(self):
        #F = xdot + x - u = 0  ->  x' = u - x  ->  x -> u
        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope
        func = lambda x, xdot, u, t: xdot + x - u
        dae = FullyImplicitDAE(func, initial_value=0.0)
        src = Constant(4.0)
        sco = Scope()
        sim = Simulation(
            blocks=[src, dae, sco],
            connections=[Connection(src, dae), Connection(dae[0], sco[0])],
            dt=0.01, Solver=ESDIRK43, log=False
            )
        sim.run(20.0)
        self.assertAlmostEqual(float(dae.engine.state[0]), 4.0, places=4)

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
