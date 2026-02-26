########################################################################################
##
##                Testing Rescale, Atan2, Alias, and discrete Delay systems
##
##   Verifies new math blocks and discrete delay mode in full simulation context.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Source,
    SinusoidalSource,
    Constant,
    Delay,
    Scope,
    )

from pathsim.blocks.math import Atan2, Rescale, Alias


# TESTCASE =============================================================================

class TestRescaleSystem(unittest.TestCase):
    """
    Test Rescale block mapping a sine wave from [-1, 1] to [0, 10].

    System: Source(sin(t)) → Rescale → Scope
    Verify: output is linearly mapped to target range
    """

    def setUp(self):

        Src = SinusoidalSource(amplitude=1.0, frequency=1.0)

        self.Rsc = Rescale(i0=-1.0, i1=1.0, o0=0.0, o1=10.0)
        self.Sco = Scope(labels=["input", "rescaled"])

        blocks = [Src, self.Rsc, self.Sco]

        connections = [
            Connection(Src, self.Rsc, self.Sco[0]),
            Connection(self.Rsc, self.Sco[1]),
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_rescale_range(self):
        """Output should be in [0, 10] for input in [-1, 1]"""

        self.Sim.run(duration=3.0, reset=True)

        time, [inp, rsc] = self.Sco.read()

        #check output stays within target range (with small tolerance)
        self.assertTrue(np.all(rsc >= -0.1), "Rescaled output below lower bound")
        self.assertTrue(np.all(rsc <= 10.1), "Rescaled output above upper bound")


    def test_rescale_linearity(self):
        """Output should be linear mapping of input"""

        self.Sim.run(duration=3.0, reset=True)

        time, [inp, rsc] = self.Sco.read()

        #expected: 5 + 5 * sin(t)
        expected = 5.0 + 5.0 * inp
        error = np.max(np.abs(rsc - expected))

        self.assertLess(error, 0.01, f"Rescale linearity error: {error:.4f}")


class TestRescaleSaturationSystem(unittest.TestCase):
    """
    Test Rescale with saturation enabled.

    System: Source(ramp) → Rescale(saturate=True) → Scope
    Verify: output is clamped to target range
    """

    def test_saturation_clamps_output(self):

        #ramp from -2 to 2 over 4 seconds, mapped [0,1] -> [0,10]
        Src = Source(lambda t: t - 2.0)
        Rsc = Rescale(i0=0.0, i1=1.0, o0=0.0, o1=10.0, saturate=True)
        Sco = Scope(labels=["input", "rescaled"])

        Sim = Simulation(
            blocks=[Src, Rsc, Sco],
            connections=[
                Connection(Src, Rsc, Sco[0]),
                Connection(Rsc, Sco[1]),
                ],
            dt=0.01,
            log=False
            )

        Sim.run(duration=4.0, reset=True)

        time, [inp, rsc] = Sco.read()

        #output should never exceed [0, 10]
        self.assertTrue(np.all(rsc >= -0.01), "Saturated output below 0")
        self.assertTrue(np.all(rsc <= 10.01), "Saturated output above 10")

        #input in valid range [0, 1] should map normally
        mask_valid = (inp >= 0.0) & (inp <= 1.0)
        if np.any(mask_valid):
            expected = 10.0 * inp[mask_valid]
            error = np.max(np.abs(rsc[mask_valid] - expected))
            self.assertLess(error, 0.1)


class TestAtan2System(unittest.TestCase):
    """
    Test Atan2 block computing the angle of a rotating vector.

    System: Source(sin(t)) → Atan2 ← Source(cos(t))
    Verify: output recovers the angle t (mod 2pi)
    """

    def setUp(self):

        self.SrcY = Source(lambda t: np.sin(t))
        self.SrcX = Source(lambda t: np.cos(t))

        self.At2 = Atan2()
        self.Sco = Scope(labels=["angle"])

        blocks = [self.SrcY, self.SrcX, self.At2, self.Sco]

        connections = [
            Connection(self.SrcY, self.At2["a"]),
            Connection(self.SrcX, self.At2["b"]),
            Connection(self.At2, self.Sco),
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_atan2_recovers_angle(self):
        """atan2(sin(t), cos(t)) should equal t for t in [0, pi)"""

        self.Sim.run(duration=3.0, reset=True)

        time, [angle] = self.Sco.read()

        #check in first half period where atan2 is monotonic
        mask = time < np.pi - 0.1
        expected = time[mask]
        actual = angle[mask]

        error = np.max(np.abs(actual - expected))
        self.assertLess(error, 0.02,
            f"Atan2 angle recovery error: {error:.4f}")


class TestAliasSystem(unittest.TestCase):
    """
    Test Alias block as a transparent pass-through.

    System: Source(sin(t)) → Alias → Scope
    Verify: output is identical to input
    """

    def test_alias_transparent(self):

        Src = SinusoidalSource(amplitude=1.0, frequency=2.0)
        Als = Alias()
        Sco = Scope(labels=["input", "alias"])

        Sim = Simulation(
            blocks=[Src, Als, Sco],
            connections=[
                Connection(Src, Als, Sco[0]),
                Connection(Als, Sco[1]),
                ],
            dt=0.01,
            log=False
            )

        Sim.run(duration=2.0, reset=True)

        time, [inp, als] = Sco.read()

        self.assertTrue(np.allclose(inp, als),
            "Alias output should be identical to input")


class TestDiscreteDelaySystem(unittest.TestCase):
    """
    Test discrete-time delay using sampling_period parameter.

    System: Source(ramp) → Delay(tau, sampling_period) → Scope
    Verify: output is a staircase-delayed version of input
    """

    def setUp(self):

        self.tau = 0.1
        self.T = 0.01

        Src = Source(lambda t: t)
        self.Dly = Delay(tau=self.tau, sampling_period=self.T)
        self.Sco = Scope(labels=["input", "delayed"])

        blocks = [Src, self.Dly, self.Sco]

        connections = [
            Connection(Src, self.Dly, self.Sco[0]),
            Connection(self.Dly, self.Sco[1]),
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.001,
            log=False
            )


    def test_discrete_delay_offset(self):
        """Delayed signal should trail input by approximately tau"""

        self.Sim.run(duration=1.0, reset=True)

        time, [inp, delayed] = self.Sco.read()

        #after initial fill (t > tau + settling), check delay offset
        mask = time > self.tau + 0.2
        t_check = time[mask]
        delayed_check = delayed[mask]

        #the delayed ramp should be approximately (t - tau)
        #with staircase quantization from sampling
        expected = t_check - self.tau
        error = np.mean(np.abs(delayed_check - expected))

        self.assertLess(error, self.T + 0.01,
            f"Discrete delay mean error: {error:.4f}")


    def test_discrete_delay_zero_initial(self):
        """Output should be zero during initial fill period"""

        self.Sim.run(duration=0.5, reset=True)

        time, [inp, delayed] = self.Sco.read()

        #during first tau seconds, output should be 0
        mask = time < self.tau * 0.5
        early_output = delayed[mask]

        self.assertTrue(np.all(early_output == 0.0),
            "Discrete delay output should be zero before buffer fills")


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
