########################################################################################
##
##                    Testing signal processing system
##
##   Tests delay, sample-hold, and filter blocks in a signal processing chain.
##   Verifies correct time delay, periodic sampling, and filtering behavior.
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    SinusoidalSource,
    Source,
    Delay,
    SampleHold,
    Scope
    )

from pathsim.solvers import RKCK54


# TESTCASE =============================================================================

class TestDelaySystem(unittest.TestCase):
    """
    Test that a delay block correctly delays a signal by tau.

    System: Source(sin(t)) → Delay(tau) → Scope
    Verify: output(t) = input(t - tau) for t >= tau
    """

    def setUp(self):

        self.tau = 0.5
        self.freq = 1.0

        Src = SinusoidalSource(amplitude=1.0, frequency=self.freq)
        Dly = Delay(tau=self.tau)
        self.Sco = Scope(labels=["input", "delayed"])

        blocks = [Src, Dly, self.Sco]

        connections = [
            Connection(Src, Dly, self.Sco[0]),
            Connection(Dly, self.Sco[1])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.001,
            log=False
            )


    def test_delay_matches_shifted_signal(self):
        """Verify delayed signal matches time-shifted input"""

        self.Sim.run(duration=5.0, reset=True)

        time, [inp, delayed] = self.Sco.read()

        #only check after delay has filled (t > tau + settling)
        mask = time > self.tau + 0.5
        t_check = time[mask]

        #expected delayed signal: sin(2*pi*freq*(t - tau))
        expected = np.sin(2 * np.pi * self.freq * (t_check - self.tau))
        actual = delayed[mask]

        #should match within reasonable tolerance
        error = np.max(np.abs(expected - actual))
        self.assertLess(error, 0.05,
            f"Delay error too large: {error:.4f}")


class TestSampleHoldSystem(unittest.TestCase):
    """
    Test that sample-hold block correctly samples at fixed intervals.

    System: Source(ramp) → SampleHold(T) → Scope
    Verify: output is piecewise constant, changing every T seconds
    """

    def setUp(self):

        self.T = 0.5  # sampling period

        Src = Source(lambda t: t)  # ramp
        self.SH = SampleHold(T=self.T)
        self.Sco = Scope(labels=["input", "sampled"])

        blocks = [Src, self.SH, self.Sco]

        connections = [
            Connection(Src, self.SH, self.Sco[0]),
            Connection(self.SH, self.Sco[1])
            ]

        self.Sim = Simulation(
            blocks,
            connections,
            dt=0.01,
            log=False
            )


    def test_sample_hold_piecewise_constant(self):
        """Verify sample-hold output is piecewise constant"""

        self.Sim.run(duration=5.0, reset=True)

        time, [inp, sampled] = self.Sco.read()

        #check that sampled output changes at sampling intervals
        #between samples, the output should be constant
        for k in range(1, 8):
            t_start = k * self.T + 0.01
            t_end = (k + 1) * self.T - 0.01
            mask = (time >= t_start) & (time <= t_end)

            if np.sum(mask) > 2:
                segment = sampled[mask]
                #within a hold period, all values should be equal
                self.assertAlmostEqual(
                    np.max(segment) - np.min(segment), 0.0, 2,
                    f"Sample-hold not constant in period {k}"
                    )


    def test_sample_hold_captures_correct_value(self):
        """Verify sample-hold captures the input value at sample times"""

        self.Sim.run(duration=3.0, reset=True)

        time, [inp, sampled] = self.Sco.read()

        #just after each sample time, the held value should match the ramp at sample time
        for k in range(1, 5):
            t_sample = k * self.T
            #find index just after sample time
            idx = np.searchsorted(time, t_sample + 0.02)
            if idx < len(sampled):
                held_value = sampled[idx]
                #held value should be close to t_sample (the ramp value at sample time)
                self.assertAlmostEqual(held_value, t_sample, 1,
                    f"Held value {held_value:.3f} != expected {t_sample:.3f}")


class TestDelayAdaptive(unittest.TestCase):
    """Test delay block with adaptive solver"""

    def test_delay_with_adaptive_solver(self):

        tau = 0.3
        Src = SinusoidalSource(amplitude=1.0, frequency=2.0)
        Dly = Delay(tau=tau)
        Sco = Scope(labels=["input", "delayed"])

        Sim = Simulation(
            blocks=[Src, Dly, Sco],
            connections=[
                Connection(Src, Dly, Sco[0]),
                Connection(Dly, Sco[1])
                ],
            Solver=RKCK54,
            tolerance_lte_abs=1e-6,
            log=False
            )

        Sim.run(duration=3.0, reset=True)

        time, [inp, delayed] = Sco.read()

        mask = time > tau + 0.5
        t_check = time[mask]
        expected = np.sin(2 * np.pi * 2.0 * (t_check - tau))
        actual = delayed[mask]

        error = np.max(np.abs(expected - actual))
        self.assertLess(error, 0.1)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
