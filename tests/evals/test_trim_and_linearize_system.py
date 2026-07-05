########################################################################################
##
##                    Testing 'Simulation.trim' and 'linearize_system'
##
##   Verifies the system-level trim (fix output values / block states and
##   solve for free unknowns) and the global linearization (assembling a
##   single '(A, B, C, D)' state-space model from an interconnected block
##   diagram), see discussion pathsim/pathsim#195.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Adder, DynamicalSystem, StateSpace, Scope
from pathsim.blocks.ctrl import PID


# TESTCASE =============================================================================

class TestTrim(unittest.TestCase):
    """
    Test 'Simulation.trim' on a nonlinear plant with a free input.

    System: dx/dt = -x^2 + u, y = x
    Unlike 'steadystate()' (which can only drive dx/dt=0 given whatever
    input is already wired), 'trim()' solves backwards for the 'Constant'
    input value that produces a desired output.
    """

    def test_trim_solves_for_free_input(self):

        plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=0.0,
            jac_dyn=lambda x, u, t: -2*x
            )
        u_src = Constant(0.0)

        Sim = Simulation(
            blocks=[u_src, plant],
            connections=[Connection(u_src, plant)],
            log=False
            )

        #target output y=2.0 -> analytically dx/dt=0 requires u=x^2=4.0
        result = Sim.trim(targets=[(plant[0], 2.0)], free=[u_src])

        self.assertTrue(result.success)
        self.assertAlmostEqual(plant.outputs[0], 2.0, 6)
        self.assertAlmostEqual(u_src.value, 4.0, 6)


class TestLinearizeSystem(unittest.TestCase):
    """
    Test 'Simulation.linearize_system' against the existing in-place
    'Simulation.linearize()' as an acceptance oracle.

    Both are built from the identical set of per-block Jacobians, so a
    PID + nonlinear plant closed loop linearized in-place and then run
    should reproduce the same small-signal trajectory as the standalone
    'StateSpace' block assembled by 'linearize_system()' and driven by
    an equivalent input perturbation.
    """

    def test_lone_state_space_roundtrip(self):
        """assembling a single StateSpace block should return its own matrices"""

        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0], [1.0]])
        C = np.array([[1.0, 1.0]])
        D = np.array([[0.5]])

        ss = StateSpace(A=A, B=B, C=C, D=D)
        Sim = Simulation(blocks=[ss], connections=[], log=False)

        result = Sim.linearize_system(inputs=[ss[0]], outputs=[ss[0]], as_block=False)

        self.assertTrue(np.allclose(result.A, A))
        self.assertTrue(np.allclose(result.B, B))
        self.assertTrue(np.allclose(result.C, C))
        self.assertTrue(np.allclose(result.D, D))
        self.assertEqual(result.state_labels, ["StateSpace_0[0]", "StateSpace_0[1]"])
        self.assertEqual(result.input_labels, ["StateSpace_0"])
        self.assertEqual(result.output_labels, ["StateSpace_0"])


    def test_pid_plant_closed_loop_matches_in_place_linearization(self):

        ref = Constant(1.0)
        err = Adder("+-")
        ctrl = PID(Kp=1.0, Ki=2.0, Kd=0.0, f_max=50)
        plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2*x
            )
        Sco = Scope(labels=["y"])

        Sim = Simulation(
            blocks=[ref, err, ctrl, plant, Sco],
            connections=[
                Connection(ref, err[0]),
                Connection(plant, err[1]),
                Connection(err, ctrl),
                Connection(ctrl, plant),
                Connection(plant, Sco),
                ],
            dt=0.01,
            log=False
            )

        #trim to the reference operating point (Ki != 0 -> y settles at ref)
        result = Sim.trim(targets=[], free=[])
        self.assertTrue(result.success)
        y0 = plant.outputs[0]
        self.assertAlmostEqual(y0, 1.0, 6)

        #baseline: existing in-place linearization + transient run
        Sim.linearize()

        #assemble the global closed-loop state-space model at the same
        #operating point, breaking the loop at the reference input
        lin_result = Sim.linearize_system(inputs=[err[0]], outputs=[plant[0]], as_block=False)
        A, B, C, D = lin_result.A, lin_result.B, lin_result.C, lin_result.D
        self.assertEqual(A.shape, (3, 3))
        self.assertEqual(B.shape, (3, 1))
        self.assertEqual(C.shape, (1, 3))
        self.assertEqual(D.shape, (1, 1))
        self.assertEqual(lin_result.input_labels, ["Adder_0"])
        self.assertEqual(lin_result.output_labels, ["DynamicalSystem_0"])

        #apply a small reference step and run the (now linear) closed loop
        delta = 0.05
        ref.value += delta
        Sim.run(duration=3.0, reset=False)
        _, (y_baseline,) = Sco.read()

        #drive the standalone assembled model with the same delta step
        #(state-space matrices are in small-signal/delta coordinates)
        step_src = Constant(delta)
        ss_block = StateSpace(A=A, B=B, C=C, D=D, initial_value=np.zeros(A.shape[0]))
        Sco2 = Scope(labels=["dy"])

        Sim2 = Simulation(
            blocks=[step_src, ss_block, Sco2],
            connections=[
                Connection(step_src, ss_block),
                Connection(ss_block, Sco2),
                ],
            dt=0.01,
            log=False
            )
        Sim2.run(duration=3.0, reset=False)
        _, (y_delta,) = Sco2.read()

        y_delta_expected = np.asarray(y_baseline) - y0
        self.assertTrue(np.allclose(y_delta_expected, y_delta, atol=1e-8))

        Sim.delinearize()


    def test_algebraic_loop_survives_break_raises(self):
        """an algebraic loop through purely algebraic blocks that is not
        broken by the marked input point should raise, rather than silently
        assemble an incorrect model"""

        #two purely algebraic blocks feeding each other directly (no
        #dynamic block breaks the algebraic simultaneity)
        from pathsim.blocks import Amplifier

        a = Amplifier(0.5)
        b = Amplifier(0.5)

        Sim = Simulation(
            blocks=[a, b],
            connections=[Connection(a, b), Connection(b, a)],
            log=False
            )

        with self.assertRaises(RuntimeError):
            Sim.linearize_system(inputs=[], outputs=[a[0]])


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
