########################################################################################
##
##            Testing 'linearize_system()' compatibility with python-control
##
##   Confirms the 'LinearizationResult' produced by 'Simulation.linearize_system()'
##   is directly usable by 'control.StateSpace', with no adapter/glue code.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import control

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Adder, DynamicalSystem, StateSpace, Scope
from pathsim.blocks.ctrl import PID


# TESTCASE =============================================================================

class TestLinearizeSystemPythonControlCompatibility(unittest.TestCase):
    """
    Build the PID + nonlinear-plant closed loop, assemble its global linear
    model, and hand it directly to 'control.StateSpace' -- no conversion
    layer should be needed.
    """

    def setUp(self):

        self.ref = Constant(1.0)
        self.err = Adder("+-")
        self.ctrl = PID(Kp=1.0, Ki=2.0, Kd=0.0, f_max=50)
        self.plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2*x
            )
        self.Sco = Scope(labels=["y"])

        self.Sim = Simulation(
            blocks=[self.ref, self.err, self.ctrl, self.plant, self.Sco],
            connections=[
                Connection(self.ref, self.err[0]),
                Connection(self.plant, self.err[1]),
                Connection(self.err, self.ctrl),
                Connection(self.ctrl, self.plant),
                Connection(self.plant, self.Sco),
                ],
            dt=0.01,
            log=False
            )

        self.Sim.trim(targets=[], free=[])
        self.result = self.Sim.linearize_system(
            inputs=[self.err[0]], outputs=[self.plant[0]], as_block=False
            )


    def test_zero_glue_construction(self):
        """the labels are exactly the kwargs 'control.StateSpace' expects"""

        sys_pc = control.StateSpace(
            self.result.A, self.result.B, self.result.C, self.result.D,
            states=self.result.state_labels,
            inputs=self.result.input_labels,
            outputs=self.result.output_labels,
            )

        self.assertEqual(list(sys_pc.state_labels), self.result.state_labels)
        self.assertEqual(list(sys_pc.input_labels), self.result.input_labels)
        self.assertEqual(list(sys_pc.output_labels), self.result.output_labels)


    def test_poles_match(self):
        """python-control's own pole computation matches pathsim's eigvals(A)"""

        sys_pc = control.StateSpace(self.result.A, self.result.B, self.result.C, self.result.D)

        poles_pc = np.sort_complex(sys_pc.poles())
        poles_ps = np.sort_complex(np.linalg.eigvals(self.result.A))

        np.testing.assert_allclose(poles_pc, poles_ps, atol=1e-8)


    def test_dc_gain(self):
        """closed-loop integral action forces unity reference-to-output DC gain"""

        sys_pc = control.StateSpace(self.result.A, self.result.B, self.result.C, self.result.D)
        self.assertAlmostEqual(float(control.dcgain(sys_pc)), 1.0, places=6)


    def test_step_response_matches_pathsim_simulation(self):
        """python-control's step response and pathsim's own simulated
        response of the identical StateSpace model agree in the time domain,
        not just in the assembled matrices"""

        delta = 0.05
        duration = 3.0

        sys_pc = control.StateSpace(self.result.A, self.result.B, self.result.C, self.result.D)
        t_pc, y_pc = control.step_response(delta * sys_pc, T=np.arange(0, duration, 0.01))

        ss_block = StateSpace(
            A=self.result.A, B=self.result.B, C=self.result.C, D=self.result.D,
            initial_value=np.zeros(self.result.A.shape[0])
            )
        step_src = Constant(delta)
        Sco2 = Scope(labels=["dy"])
        Sim2 = Simulation(
            blocks=[step_src, ss_block, Sco2],
            connections=[Connection(step_src, ss_block), Connection(ss_block, Sco2)],
            dt=0.01,
            log=False
            )
        Sim2.run(duration=duration, reset=False)
        t_ps, (y_ps,) = Sco2.read()

        y_pc_interp = np.interp(t_ps, t_pc, y_pc)
        self.assertTrue(np.max(np.abs(y_ps - y_pc_interp)) < 1e-3)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
