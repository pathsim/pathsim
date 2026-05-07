########################################################################################
##
##                                  TESTS FOR
##                              'blocks.discrete.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.discrete import (
    SampleHold,
    ZeroOrderHold,
    FirstOrderHold,
    FIR,
    DiscreteIntegrator,
    DiscreteDerivative,
    DiscreteStateSpace,
    DiscreteTransferFunction,
    TappedDelay,
    )
from pathsim.events.schedule import Schedule


# SAMPLE & HOLD ========================================================================

class TestSampleHold(unittest.TestCase):

    def test_init(self):
        SH = SampleHold()
        self.assertEqual(SH.T, 1.0)
        self.assertEqual(SH.tau, 0.0)
        self.assertEqual(len(SH.events), 1)
        self.assertIsInstance(SH.events[0], Schedule)

        SH = SampleHold(T=0.5, tau=0.1)
        self.assertEqual(SH.T, 0.5)
        self.assertEqual(SH.tau, 0.1)


    def test_len(self):
        self.assertEqual(len(SampleHold()), 0)


    def test_event_scheduling(self):
        SH = SampleHold(T=2.0, tau=0.5)
        self.assertEqual(SH.events[0].t_start, 0.5)
        self.assertEqual(SH.events[0].t_period, 2.0)


    def test_sample_and_hold(self):
        SH = SampleHold(T=1.0)
        SH.inputs[0] = 5.0
        SH.events[0].func_act(0)
        self.assertEqual(SH.outputs[0], 5.0)

        #change input without sampling -> output holds
        SH.inputs[0] = 20.0
        self.assertEqual(SH.outputs[0], 5.0)

        #next sample picks up new value
        SH.events[0].func_act(1)
        self.assertEqual(SH.outputs[0], 20.0)


    def test_zero_order_hold_alias(self):
        self.assertIs(ZeroOrderHold, SampleHold)


    def test_vector_input(self):
        SH = SampleHold(T=1.0)
        SH.inputs[0] = 1.0
        SH.inputs[1] = 2.0
        SH.inputs[2] = 3.0
        SH.events[0].func_act(0)
        np.testing.assert_array_equal(SH.outputs.to_array(), [1.0, 2.0, 3.0])

        SH.inputs[0] = 10.0
        SH.inputs[1] = 20.0
        SH.inputs[2] = 30.0
        SH.events[0].func_act(1)
        np.testing.assert_array_equal(SH.outputs.to_array(), [10.0, 20.0, 30.0])


# FIRST-ORDER HOLD =====================================================================

class TestFirstOrderHold(unittest.TestCase):

    def test_init(self):
        FOH = FirstOrderHold(T=0.5, tau=0.1)
        self.assertEqual(FOH.T, 0.5)
        self.assertEqual(FOH.tau, 0.1)
        self.assertEqual(len(FOH.events), 1)


    def test_len(self):
        self.assertEqual(len(FirstOrderHold()), 0)


    def test_hold_during_first_interval(self):
        """before two samples are captured, output holds the latest value"""
        FOH = FirstOrderHold(T=1.0)
        FOH.inputs[0] = 2.0
        FOH.events[0].func_act(0.0)
        FOH.update(0.5)
        self.assertEqual(FOH.outputs[0], 2.0)


    def test_linear_extrapolation(self):
        """after two samples, output extrapolates linearly with previous slope"""
        FOH = FirstOrderHold(T=1.0)

        FOH.inputs[0] = 1.0
        FOH.events[0].func_act(0.0)
        FOH.inputs[0] = 3.0
        FOH.events[0].func_act(1.0)
        #slope = (3 - 1) / 1 = 2, extrapolated from t=1
        FOH.update(1.0)
        self.assertAlmostEqual(FOH.outputs[0], 3.0)
        FOH.update(1.5)
        self.assertAlmostEqual(FOH.outputs[0], 4.0)
        FOH.update(2.0)
        self.assertAlmostEqual(FOH.outputs[0], 5.0)


    def test_reset(self):
        FOH = FirstOrderHold(T=1.0)
        FOH.inputs[0] = 5.0
        FOH.events[0].func_act(0.0)
        FOH.events[0].func_act(1.0)
        FOH.reset()
        FOH.update(0.5)
        self.assertEqual(FOH.outputs[0], 0.0)


    def test_vector_input(self):
        FOH = FirstOrderHold(T=1.0)

        FOH.inputs[0] = 1.0
        FOH.inputs[1] = 10.0
        FOH.events[0].func_act(0.0)
        FOH.inputs[0] = 3.0
        FOH.inputs[1] = 20.0
        FOH.events[0].func_act(1.0)
        #slopes per channel: [2.0, 10.0]
        FOH.update(1.5)
        np.testing.assert_allclose(FOH.outputs.to_array(), [4.0, 25.0])


# FIR =================================================================================

class TestFIR(unittest.TestCase):

    def test_init(self):
        F = FIR()
        np.testing.assert_array_equal(F.coeffs, [1.0])
        self.assertEqual(F.T, 1.0)
        self.assertEqual(F.tau, 0.0)

        F = FIR(coeffs=[0.5, 0.3, 0.2], T=0.1, tau=0.05)
        np.testing.assert_array_equal(F.coeffs, [0.5, 0.3, 0.2])
        self.assertEqual(F.T, 0.1)
        self.assertEqual(F.tau, 0.05)


    def test_len(self):
        self.assertEqual(len(FIR()), 0)


    def test_passthrough(self):
        F = FIR(coeffs=[1.0])
        F.inputs[0] = 5.0
        F.events[0].func_act(0)
        self.assertEqual(F.outputs[0], 5.0)


    def test_moving_average(self):
        F = FIR(coeffs=[1/3, 1/3, 1/3])
        for u, expected in [(3.0, 1.0), (6.0, 3.0), (9.0, 6.0)]:
            F.inputs[0] = u
            F.events[0].func_act(0)
            self.assertAlmostEqual(F.outputs[0], expected, places=10)


    def test_difference_filter(self):
        F = FIR(coeffs=[1.0, -1.0])
        for u, expected in [(5.0, 5.0), (8.0, 3.0), (10.0, 2.0)]:
            F.inputs[0] = u
            F.events[0].func_act(0)
            self.assertEqual(F.outputs[0], expected)


    def test_reset(self):
        F = FIR(coeffs=[1.0, 0.5])
        F.inputs[0] = 10.0
        F.events[0].func_act(0)
        F.reset()
        for val in F._buffer:
            self.assertEqual(val, 0.0)


    def test_vector_input(self):
        """same coefficients applied to each channel in parallel"""
        F = FIR(coeffs=[1.0, 0.5])

        F.inputs[0] = 2.0
        F.inputs[1] = 4.0
        F.events[0].func_act(0)
        np.testing.assert_allclose(F.outputs.to_array(), [2.0, 4.0])

        F.inputs[0] = 6.0
        F.inputs[1] = 8.0
        F.events[0].func_act(0)
        #y = 1*[6,8] + 0.5*[2,4] = [7, 10]
        np.testing.assert_allclose(F.outputs.to_array(), [7.0, 10.0])


# DISCRETE INTEGRATOR =================================================================

class TestDiscreteIntegrator(unittest.TestCase):

    def test_init(self):
        DI = DiscreteIntegrator(T=0.5, initial_value=2.0)
        self.assertEqual(DI.T, 0.5)
        self.assertEqual(DI.initial_value, 2.0)
        self.assertEqual(DI.outputs[0], 2.0)


    def test_len(self):
        self.assertEqual(len(DiscreteIntegrator()), 0)


    def test_forward_euler(self):
        """y[k+1] = y[k] + T * u[k]"""
        DI = DiscreteIntegrator(T=0.5)

        #step 0: output IC, advance with u=2 -> next state = 0 + 0.5*2 = 1
        DI.inputs[0] = 2.0
        DI.events[0].func_act(0.0)
        self.assertEqual(DI.outputs[0], 0.0)

        #step 1: output 1, advance with u=4 -> next = 1 + 0.5*4 = 3
        DI.inputs[0] = 4.0
        DI.events[0].func_act(0.5)
        self.assertEqual(DI.outputs[0], 1.0)

        #step 2: output 3
        DI.inputs[0] = 0.0
        DI.events[0].func_act(1.0)
        self.assertEqual(DI.outputs[0], 3.0)


    def test_initial_value(self):
        DI = DiscreteIntegrator(T=1.0, initial_value=5.0)
        DI.inputs[0] = 1.0
        DI.events[0].func_act(0.0)
        self.assertEqual(DI.outputs[0], 5.0)
        DI.events[0].func_act(1.0)
        self.assertEqual(DI.outputs[0], 6.0)


    def test_reset(self):
        DI = DiscreteIntegrator(T=1.0, initial_value=2.0)
        DI.inputs[0] = 5.0
        DI.events[0].func_act(0.0)
        DI.events[0].func_act(1.0)
        DI.reset()
        self.assertEqual(DI.outputs[0], 2.0)


    def test_vector_input(self):
        DI = DiscreteIntegrator(T=0.5, initial_value=[1.0, -1.0])

        np.testing.assert_array_equal(DI.outputs.to_array(), [1.0, -1.0])

        DI.inputs[0] = 2.0
        DI.inputs[1] = 4.0
        DI.events[0].func_act(0.0)
        np.testing.assert_array_equal(DI.outputs.to_array(), [1.0, -1.0])

        DI.inputs[0] = 0.0
        DI.inputs[1] = 0.0
        DI.events[0].func_act(0.5)
        #state advanced to [1+0.5*2, -1+0.5*4] = [2, 1]
        np.testing.assert_allclose(DI.outputs.to_array(), [2.0, 1.0])


# DISCRETE DERIVATIVE =================================================================

class TestDiscreteDerivative(unittest.TestCase):

    def test_backward_difference(self):
        """y[k] = (u[k] - u[k-1]) / T"""
        DD = DiscreteDerivative(T=0.5)

        DD.inputs[0] = 2.0
        DD.events[0].func_act(0.0)
        self.assertEqual(DD.outputs[0], 4.0)  # (2 - 0)/0.5

        DD.inputs[0] = 5.0
        DD.events[0].func_act(0.5)
        self.assertEqual(DD.outputs[0], 6.0)  # (5 - 2)/0.5

        DD.inputs[0] = 5.0
        DD.events[0].func_act(1.0)
        self.assertEqual(DD.outputs[0], 0.0)  # (5 - 5)/0.5


    def test_len(self):
        self.assertEqual(len(DiscreteDerivative()), 0)


    def test_reset(self):
        DD = DiscreteDerivative(T=1.0)
        DD.inputs[0] = 10.0
        DD.events[0].func_act(0.0)
        DD.reset()
        DD.inputs[0] = 3.0
        DD.events[0].func_act(1.0)
        self.assertEqual(DD.outputs[0], 3.0)


    def test_vector_input(self):
        DD = DiscreteDerivative(T=0.5)

        DD.inputs[0] = 2.0
        DD.inputs[1] = 4.0
        DD.events[0].func_act(0.0)
        np.testing.assert_allclose(DD.outputs.to_array(), [4.0, 8.0])

        DD.inputs[0] = 5.0
        DD.inputs[1] = 5.0
        DD.events[0].func_act(0.5)
        np.testing.assert_allclose(DD.outputs.to_array(), [6.0, 2.0])


# DISCRETE STATE SPACE ================================================================

class TestDiscreteStateSpace(unittest.TestCase):

    def test_scalar_init(self):
        DSS = DiscreteStateSpace(A=0.5, B=1.0, C=2.0, D=0.0, T=0.1)
        self.assertEqual(len(DSS.inputs), 1)
        self.assertEqual(len(DSS.outputs), 1)


    def test_scalar_dynamics(self):
        """x[k+1] = 0.5 x[k] + u[k], y[k] = x[k]"""
        DSS = DiscreteStateSpace(A=[[0.5]], B=[[1.0]], C=[[1.0]], D=[[0.0]],
                                 T=1.0, initial_value=[0.0])

        DSS.inputs[0] = 1.0
        DSS.events[0].func_act(0.0)
        self.assertAlmostEqual(DSS.outputs[0], 0.0)

        DSS.inputs[0] = 1.0
        DSS.events[0].func_act(1.0)
        self.assertAlmostEqual(DSS.outputs[0], 1.0)

        DSS.inputs[0] = 1.0
        DSS.events[0].func_act(2.0)
        self.assertAlmostEqual(DSS.outputs[0], 1.5)


    def test_direct_feedthrough(self):
        """y[k] = D u[k] when C is zero"""
        DSS = DiscreteStateSpace(A=[[0.0]], B=[[0.0]], C=[[0.0]], D=[[3.0]],
                                 T=1.0)
        DSS.inputs[0] = 2.0
        DSS.events[0].func_act(0.0)
        self.assertAlmostEqual(DSS.outputs[0], 6.0)


    def test_mimo(self):
        """2-state MIMO: 2 inputs, 2 outputs"""
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.eye(2)
        D = np.zeros((2, 2))
        DSS = DiscreteStateSpace(A=A, B=B, C=C, D=D, T=1.0,
                                 initial_value=[0.0, 0.0])
        self.assertEqual(len(DSS.inputs), 2)
        self.assertEqual(len(DSS.outputs), 2)

        DSS.inputs[0] = 1.0
        DSS.inputs[1] = 2.0
        DSS.events[0].func_act(0.0)
        np.testing.assert_allclose(DSS.outputs.to_array(), [0.0, 0.0])

        DSS.events[0].func_act(1.0)
        #x[1] = A*0 + B*[1,2] = [1,2]; y[1] = [1,2]
        np.testing.assert_allclose(DSS.outputs.to_array(), [1.0, 2.0])


    def test_reset(self):
        DSS = DiscreteStateSpace(A=[[1.0]], B=[[1.0]], C=[[1.0]], D=[[0.0]],
                                 T=1.0, initial_value=[5.0])
        DSS.inputs[0] = 1.0
        DSS.events[0].func_act(0.0)
        DSS.events[0].func_act(1.0)
        DSS.reset()
        np.testing.assert_array_equal(DSS._x, [5.0])


# DISCRETE TRANSFER FUNCTION ==========================================================

class TestDiscreteTransferFunction(unittest.TestCase):

    def test_init(self):
        DTF = DiscreteTransferFunction(Num=[1.0], Den=[1.0, -0.5], T=0.1)
        self.assertEqual(DTF.T, 0.1)


    def test_first_order(self):
        """H(z) = 1/(z - 0.5)  ->  y[k+1] = 0.5 y[k] + u[k]"""
        DTF = DiscreteTransferFunction(Num=[1.0], Den=[1.0, -0.5], T=1.0)

        DTF.inputs[0] = 1.0
        DTF.events[0].func_act(0.0)
        y0 = DTF.outputs[0]

        DTF.inputs[0] = 1.0
        DTF.events[0].func_act(1.0)
        y1 = DTF.outputs[0]

        DTF.inputs[0] = 1.0
        DTF.events[0].func_act(2.0)
        y2 = DTF.outputs[0]

        #recurrence: y[0]=0, y[1]=1, y[2]=0.5*1+1=1.5
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(y1, 1.0)
        self.assertAlmostEqual(y2, 1.5)


# TAPPED DELAY ========================================================================

class TestTappedDelay(unittest.TestCase):

    def test_init(self):
        TD = TappedDelay(N=4, T=0.5)
        self.assertEqual(TD.N, 4)
        self.assertEqual(len(TD.outputs), 4)


    def test_len(self):
        self.assertEqual(len(TappedDelay()), 0)


    def test_shifts(self):
        TD = TappedDelay(N=3, T=1.0)

        TD.inputs[0] = 1.0
        TD.events[0].func_act(0.0)
        np.testing.assert_array_equal(TD.outputs.to_array(), [1.0, 0.0, 0.0])

        TD.inputs[0] = 2.0
        TD.events[0].func_act(1.0)
        np.testing.assert_array_equal(TD.outputs.to_array(), [2.0, 1.0, 0.0])

        TD.inputs[0] = 3.0
        TD.events[0].func_act(2.0)
        np.testing.assert_array_equal(TD.outputs.to_array(), [3.0, 2.0, 1.0])

        TD.inputs[0] = 4.0
        TD.events[0].func_act(3.0)
        np.testing.assert_array_equal(TD.outputs.to_array(), [4.0, 3.0, 2.0])


    def test_reset(self):
        TD = TappedDelay(N=2, T=1.0)
        TD.inputs[0] = 7.0
        TD.events[0].func_act(0.0)
        TD.reset()
        for val in TD._buffer:
            self.assertEqual(val, 0.0)


# RUN ==================================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
